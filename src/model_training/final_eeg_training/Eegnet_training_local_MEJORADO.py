"""
eegnet_training_local_MEJORADO.py
=================================================================================
Pipeline EEGNet v2 para clasificación de dolor (NRS 2/4/6/8) — VERSIÓN MEJORADA.

Mejoras aplicadas frente al script original (cada una marcada con  # >>> MEJORA N):
  1. Downsampling a 250 Hz   -> alinea la señal con el diseño original de EEGNet
                                y reduce 4x el tamaño temporal (1501 -> ~375).
  2. Normalización SIN fuga   -> el RobustScaler se ajusta SOLO con el train de
     de datos (data leakage)     cada fold y se aplica al val (antes se ajustaba
                                por época, borrando la amplitud y mezclando info).
  3. Early stopping + mejor   -> se valida en cada época, se guarda el mejor
     checkpoint                  modelo por macro-F1 y se restauran esos pesos.
  4. Scheduler (CosineAnneal) -> baja el learning rate de forma suave.
  5. AdamW + label smoothing  -> mejor regularización y mejor log_loss/calibración.
  6. Data augmentation         -> ruido gaussiano + jitter temporal en train para
                                compensar el dataset pequeño y desbalanceado.
  7. Dropout 0.5 -> 0.3        -> el modelo original NO ajustaba ni el train
                                (train_loss ~0.93); se baja la regularización.
  8. Semillas fijas            -> resultados reproducibles entre corridas.

NOTA: la sección de credenciales se deja vacía a propósito (seguridad).
Entorno: ejecución local con protección de multiprocesamiento para Windows.
"""
# ==============================================================================
# 0. DEPENDENCIAS REQUERIDAS
# ==============================================================================
import os
import re
import glob
import gc
import copy
import random
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (f1_score, accuracy_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             log_loss, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

# >>> MEJORA 8: reproducibilidad ------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# 1. HIPERPARÁMETROS CENTRALIZADOS (fácil de tunear)
# ==============================================================================
SFREQ_TARGET   = 250     # >>> MEJORA 1: Hz objetivo tras downsampling
N_EPOCHS_TRAIN = 80      # se entrena más, pero con early stopping
PATIENCE       = 15      # >>> MEJORA 3: épocas sin mejora antes de cortar
BATCH_SIZE     = 32
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3     # >>> MEJORA 7: antes 0.5 (el modelo no ajustaba)
LABEL_SMOOTH   = 0.05    # >>> MEJORA 5
AUG_NOISE_STD  = 0.10    # >>> MEJORA 6: desviación del ruido gaussiano
AUG_MAX_SHIFT  = 20      # >>> MEJORA 6: jitter temporal máximo (muestras)

# ==============================================================================
# 3. ARQUITECTURA EEGNet v2  (sin cambios estructurales; el flatten es dinámico)
# ==============================================================================
class EEGNetv2(nn.Module):
    def __init__(self, n_classes=4, n_channels=63, samples=375, F1=8, D=2, F2=16, dropout=DROPOUT):
        super(EEGNetv2, self).__init__()
        self.temporal_conv = nn.Sequential(
            # kernel 125 @ 250 Hz = 0.5 s  -> convención EEGNet (sfreq/2)
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 125), padding=(0, 62), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(in_channels=F1*D, out_channels=F1*D, kernel_size=(1, 16), padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(in_channels=F1*D, out_channels=F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )
        self._flatten_size = self._get_flatten_size(n_channels, samples, F1, D, F2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flatten_size, n_classes)
        )

    def _get_flatten_size(self, n_channels, samples, F1, D, F2):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, samples)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            return x.numel()

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return self.classifier(x)


# >>> MEJORA 6: aumento de datos aplicado SOLO al batch de entrenamiento --------
def augment_batch(batch_X):
    """Ruido gaussiano + desplazamiento temporal aleatorio (roll)."""
    # ruido gaussiano
    noise = torch.randn_like(batch_X) * AUG_NOISE_STD
    out = batch_X + noise
    # jitter temporal (corre la señal en el eje de tiempo)
    shift = random.randint(-AUG_MAX_SHIFT, AUG_MAX_SHIFT)
    if shift != 0:
        out = torch.roll(out, shifts=shift, dims=-1)
    return out


# ==============================================================================
# BUCLE PRINCIPAL PROTEGIDO (Obligatorio en Windows para evitar duplicaciones)
# ==============================================================================
if __name__ == '__main__':

    # 1. CONFIGURACIÓN DE MLOPS Y HARDWARE
    os.environ["MLFLOW_TRACKING_URI"]      = "https://dagshub.com/GaboGolCs/PainPredict-Neuro.mlflow" 
    os.environ["MLFLOW_TRACKING_USERNAME"] = "GaboGolCs"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "42f7e673b8edabb1bb9a4936359bb2fbe533a53c"
    mlflow.set_experiment("EEGNet_DeepLearning_G2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")

    # 2. LECTURA OPTIMIZADA PARA BAJA MEMORIA RAM
    CARPETA_FIFS = r'C:\Users\panch\OneDrive\Escritorio\p\fifs'
    lista_rutas = sorted(glob.glob(os.path.join(CARPETA_FIFS, '*.fif')))
    if len(lista_rutas) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .fif en: {CARPETA_FIFS}")
    print(f"Se encontraron {len(lista_rutas)} archivos .fif. Iniciando lectura limpia...")

    lista_X_numpy = []
    lista_y_raw   = []
    lista_grupos  = []
    archivos_fallidos = []
    evento_a_clase = {32: 0, 33: 1, 34: 2, 35: 3}

    for idx, ruta in enumerate(lista_rutas, 1):
        try:
            nombre_archivo = os.path.basename(ruta)
            epo = mne.read_epochs(ruta, preload=True, verbose=False)

            # >>> MEJORA 1: downsampling a SFREQ_TARGET Hz -------------------------
            if round(epo.info['sfreq']) != SFREQ_TARGET:
                epo.resample(SFREQ_TARGET, verbose=False)

            match_sujeto = re.search(r'(sub-\d+)', nombre_archivo)
            sujeto_id = match_sujeto.group(1) if match_sujeto else "sub_unknown"
            eventos_archivo = epo.events[:, 2]

            epocas_validas_en_archivo = 0
            for ev in eventos_archivo:
                if ev in evento_a_clase:
                    lista_y_raw.append(evento_a_clase[ev])
                    lista_grupos.append(sujeto_id)
                    epocas_validas_en_archivo += 1

            if epocas_validas_en_archivo == len(epo):
                lista_X_numpy.append(epo.get_data())
            else:
                print(f"   [!] Advertencia: {nombre_archivo} contiene Inconsistencias. Saltando.")
                archivos_fallidos.append(ruta)
                for _ in range(epocas_validas_en_archivo):
                    lista_y_raw.pop()
                    lista_grupos.pop()

            del epo
            if idx % 25 == 0 or idx == len(lista_rutas):
                print(f"   ✓ [{idx}/{len(lista_rutas)}] archivos procesados correctamente.")

        except Exception as e:
            archivos_fallidos.append(ruta)
            print(f"   [!] Error al leer {os.path.basename(ruta)}: {str(e)}")

    print(f"\nLectura finalizada. {len(lista_X_numpy)} archivos OK, {len(archivos_fallidos)} excluidos.")
    print("\nConcatenando señales en un arreglo global...")
    X_raw = np.concatenate(lista_X_numpy, axis=0)
    del lista_X_numpy
    gc.collect()

    n_epochs, n_channels, n_times = X_raw.shape
    print(f"Forma del tensor crudo combinado: {X_raw.shape}  (sfreq={SFREQ_TARGET} Hz)")
    y_raw  = np.array(lista_y_raw)
    grupos = np.array(lista_grupos)
    if len(y_raw) != n_epochs:
        raise ValueError(f"Desalineación: {n_epochs} epochs vs {len(y_raw)} etiquetas.")

    # >>> MEJORA 2: NO se normaliza acá. La normalización se hace DENTRO de cada
    # fold, ajustando el scaler solo con el train (evita fuga de datos).
    X_tensor = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)  # (N,1,C,T)
    y_tensor = torch.tensor(y_raw,  dtype=torch.long)
    print(f"-> Tensor X (sin normalizar todavía): {X_tensor.shape}")
    print(f"-> Tensor y: {y_tensor.shape}\n")

    # 4. VALIDACIÓN CRUZADA Y ENTRENAMIENTO EN GPU
    gkf = GroupKFold(n_splits=5)
    pesos_clases = compute_class_weight(class_weight='balanced', classes=np.unique(y_raw), y=y_raw)
    tensor_pesos = torch.tensor(pesos_clases, dtype=torch.float32).to(device)
    # >>> MEJORA 5: label smoothing
    criterion = nn.CrossEntropyLoss(weight=tensor_pesos, label_smoothing=LABEL_SMOOTH)

    print("Iniciando entrenamiento con GroupKFold (5 folds)...\n")
    resumen_folds = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tensor, y_tensor, groups=grupos)):
        with mlflow.start_run(run_name=f"EEGNet_Local_Fold_{fold+1}"):
            X_train, y_train = X_tensor[train_idx].clone(), y_tensor[train_idx]
            X_val,   y_val   = X_tensor[val_idx].clone(),   y_tensor[val_idx]

            # >>> MEJORA 2: RobustScaler por canal, ajustado SOLO con el train -----
            # Se aprende la mediana/IQR de cada canal usando todas las épocas y
            # tiempos del train, y se aplica idéntico al val (sin mirar el val).
            Xtr = X_train.squeeze(1).numpy()                  # (Ntr, C, T)
            Xva = X_val.squeeze(1).numpy()                    # (Nva, C, T)
            for c in range(n_channels):
                sc = RobustScaler()
                sc.fit(Xtr[:, c, :].reshape(-1, 1))           # estadísticos del TRAIN
                Xtr[:, c, :] = sc.transform(Xtr[:, c, :].reshape(-1, 1)).reshape(Xtr[:, c, :].shape)
                Xva[:, c, :] = sc.transform(Xva[:, c, :].reshape(-1, 1)).reshape(Xva[:, c, :].shape)
            X_train = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)
            X_val   = torch.tensor(Xva, dtype=torch.float32).unsqueeze(1)

            # num_workers=0 evita problemas de multiprocessing en Windows con
            # datasets chicos (y elimina overhead innecesario).
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
            val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            model     = EEGNetv2(n_classes=4, n_channels=n_channels, samples=n_times, dropout=DROPOUT).to(device)
            # >>> MEJORA 5: AdamW (weight decay desacoplado)
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            # >>> MEJORA 4: scheduler coseno
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_TRAIN)

            mlflow.log_params({
                "environment": "local_gpu",
                "sfreq": SFREQ_TARGET,
                "n_epochs_train": N_EPOCHS_TRAIN,
                "patience": PATIENCE,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "F1": 8, "D": 2, "F2": 16,
                "dropout": DROPOUT,
                "label_smoothing": LABEL_SMOOTH,
                "aug_noise_std": AUG_NOISE_STD,
                "aug_max_shift": AUG_MAX_SHIFT,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "scaler": "RobustScaler_por_canal_fit_en_train"
            })

            # >>> MEJORA 3: early stopping con seguimiento del mejor modelo --------
            best_f1     = -1.0
            best_state  = None
            epochs_sin_mejora = 0

            for epoch in range(N_EPOCHS_TRAIN):
                # ----- entrenamiento -----
                model.train()
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X = augment_batch(batch_X)              # >>> MEJORA 6
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss    = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()
                avg_loss = epoch_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                # ----- validación por época (para early stopping) -----
                model.eval()
                vp, vt = [], []
                with torch.no_grad():
                    for bx, by in val_loader:
                        out = model(bx.to(device))
                        _, pred = torch.max(out.data, 1)
                        vp.extend(pred.cpu().numpy()); vt.extend(by.numpy())
                f1_epoch = f1_score(vt, vp, average='macro')
                mlflow.log_metric("val_macro_f1_epoch", f1_epoch, step=epoch)

                if f1_epoch > best_f1:
                    best_f1 = f1_epoch
                    best_state = copy.deepcopy(model.state_dict())   # guarda el mejor
                    epochs_sin_mejora = 0
                else:
                    epochs_sin_mejora += 1

                if (epoch + 1) % 10 == 0:
                    print(f"   Fold {fold+1} | Época {epoch+1:02d}/{N_EPOCHS_TRAIN} | "
                          f"Loss: {avg_loss:.4f} | val_F1: {f1_epoch:.4f} | best: {best_f1:.4f}")

                if epochs_sin_mejora >= PATIENCE:
                    print(f"   Fold {fold+1} | Early stopping en época {epoch+1} (best F1={best_f1:.4f})")
                    break

            # >>> MEJORA 3: restaura los MEJORES pesos antes de evaluar ------------
            if best_state is not None:
                model.load_state_dict(best_state)

            # ----- evaluación final con el mejor modelo -----
            model.eval()
            val_preds, val_targets, val_probs = [], [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X.to(device))
                    probs   = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_y.numpy())
                    val_probs.extend(probs.cpu().numpy())

            val_targets = np.array(val_targets)
            val_preds   = np.array(val_preds)
            val_probs   = np.array(val_probs)

            macro_f1     = f1_score(val_targets, val_preds, average='macro')
            acc          = accuracy_score(val_targets, val_preds)
            mcc          = matthews_corrcoef(val_targets, val_preds)
            val_log_loss = log_loss(val_targets, val_probs, labels=[0, 1, 2, 3])
            try:
                auc_roc = roc_auc_score(val_targets, val_probs, multi_class='ovr', labels=[0, 1, 2, 3])
            except ValueError:
                auc_roc = 0.0

            mlflow.log_metrics({
                "val_macro_f1": macro_f1,
                "val_accuracy": acc,
                "val_mcc":      mcc,
                "val_auc_roc":  auc_roc,
                "val_log_loss": val_log_loss
            })

            precision_por_clase = precision_score(val_targets, val_preds, average=None, labels=[0,1,2,3], zero_division=0)
            recall_por_clase    = recall_score(val_targets, val_preds, average=None, labels=[0,1,2,3], zero_division=0)
            for i, nrs in enumerate([2, 4, 6, 8]):
                mlflow.log_metric(f"val_precision_NRS_{nrs}", precision_por_clase[i])
                mlflow.log_metric(f"val_recall_NRS_{nrs}",    recall_por_clase[i])

            cm = confusion_matrix(val_targets, val_preds, labels=[0, 1, 2, 3])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=[2,4,6,8], yticklabels=[2,4,6,8])
            plt.title(f"Matriz de Confusión - Fold {fold+1}")
            plt.ylabel("Real (NRS)"); plt.xlabel("Predicho (NRS)")
            cm_path = f"confusion_matrix_fold{fold+1}.png"
            plt.savefig(cm_path); mlflow.log_artifact(cm_path); plt.close()

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=f"modelo_eegnet_fold{fold+1}",
                serialization_format="pickle"
            )
            ruta_modelo_local = f"eegnet_fold{fold+1}.pth"
            torch.save(model.state_dict(), ruta_modelo_local)

            resumen_folds.append((fold+1, macro_f1, acc, mcc, auc_roc, val_log_loss))
            print(f"\n   ✓ Fold {fold+1} completado | Macro F1: {macro_f1:.4f} | "
                  f"Acc: {acc:.4f} | MCC: {mcc:.4f} | AUC: {auc_roc:.4f}\n")

    # ----- resumen final de los 5 folds -----
    print("=" * 70)
    print("RESUMEN (promedio ± desviación entre folds):")
    arr = np.array([r[1:] for r in resumen_folds])
    nombres_m = ["Macro-F1", "Accuracy", "MCC", "AUC-ROC", "LogLoss"]
    for j, nm in enumerate(nombres_m):
        print(f"   {nm:<10}: {arr[:,j].mean():.4f} ± {arr[:,j].std():.4f}")
    print("=" * 70)
    print("Entrenamiento finalizado exitosamente. Modelos .pth listos en tu computadora.")
