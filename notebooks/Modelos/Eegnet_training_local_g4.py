"""
eegnet_training_local.py
Pipeline completo de entrenamiento EEGNet v2 para clasificación de dolor (NRS 2/4/6/8).
Arquitectura: Temporal Conv → DepthwiseConv → SeparableConv → Classifier
Validación clínica: GroupKFold por sujeto + CrossEntropyLoss ponderada.

Entorno: Execution Local con protección de multiprocesamiento para Windows.
"""

# ==============================================================================
# 0. DEPENDENCIAS REQUERIDAS
# ==============================================================================
import os
import re
import glob
import gc
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

# ==============================================================================
# 3. ARQUITECTURA EEGNet v2
# ==============================================================================
class EEGNetv2(nn.Module):
    def __init__(self, n_classes=4, n_channels=63, samples=1501, F1=8, D=2, F2=16, dropout=0.5):
        super(EEGNetv2, self).__init__()

        self.temporal_conv = nn.Sequential(
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

# ==============================================================================
# BUCILE PRINCIPAL PROTEGIDO (Obligatorio en Windows para evitar duplicaciones)
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
    lista_y_raw = []
    lista_grupos = []
    archivos_fallidos = []
    evento_a_clase = {32: 0, 33: 1, 34: 2, 35: 3}

    for idx, ruta in enumerate(lista_rutas, 1):
        try:
            nombre_archivo = os.path.basename(ruta)
            epo = mne.read_epochs(ruta, preload=True, verbose=False)
            
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
    print(f"Forma del tensor crudo combinado: {X_raw.shape}")

    y_raw = np.array(lista_y_raw)
    grupos = np.array(lista_grupos)

    if len(y_raw) != n_epochs:
        raise ValueError(f"Desalineación: {n_epochs} epochs vs {len(y_raw)} etiquetas.")

    print("Normalizando señales canal a canal (RobustScaler)...")
    X_norm = np.zeros_like(X_raw)
    scaler = RobustScaler()
    for i in range(n_epochs):
        X_norm[i] = scaler.fit_transform(X_raw[i].T).T

    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    print(f"-> Tensor X listo para la GPU: {X_tensor.shape}")
    print(f"-> Tensor y listo para la GPU: {y_tensor.shape}\n")

    # 4. VALIDACIÓN CRUZADA Y ENTRENAMIENTO EN GPU
    gkf = GroupKFold(n_splits=5)

    pesos_clases = compute_class_weight(class_weight='balanced', classes=np.unique(y_raw), y=y_raw)
    tensor_pesos = torch.tensor(pesos_clases, dtype=torch.float32).to(device)
    criterion    = nn.CrossEntropyLoss(weight=tensor_pesos)

    print("Iniciando entrenamiento acelerado por hardware con GroupKFold (5 folds)...\n")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tensor, y_tensor, groups=grupos)):
        with mlflow.start_run(run_name=f"EEGNet_Local_Fold_{fold+1}"):

            X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
            X_val,   y_val   = X_tensor[val_idx],   y_tensor[val_idx]

            # Multiprocesamiento protegido para Windows seguro
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
            val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

            model     = EEGNetv2(n_classes=4, n_channels=n_channels, samples=n_times).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

            mlflow.log_params({
                "environment": "local_gpu",
                "n_epochs_train": 50,
                "batch_size": 32,
                "lr": 0.001,
                "weight_decay": 1e-4,
                "F1": 8, "D": 2, "F2": 16,
                "dropout": 0.5
            })

            n_epochs_train = 50
            for epoch in range(n_epochs_train):
                model.train()
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss    = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                if (epoch + 1) % 10 == 0:
                    print(f"   Fold {fold+1} | Época {epoch+1:02d}/{n_epochs_train} | Loss: {avg_loss:.4f}")

            model.eval()
            val_preds, val_targets, val_probs = [], [], []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X   = batch_X.to(device)
                    outputs   = model(batch_X)
                    probs     = F.softmax(outputs, dim=1)
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
                "val_macro_f1" : macro_f1,
                "val_accuracy" : acc,
                "val_mcc"      : mcc,
                "val_auc_roc"  : auc_roc,
                "val_log_loss" : val_log_loss
            })

            precision_por_clase = precision_score(val_targets, val_preds, average=None, labels=[0,1,2,3], zero_division=0)
            recall_por_clase    = recall_score(val_targets, val_preds, average=None, labels=[0,1,2,3], zero_division=0)

            for idx, nrs in enumerate([2, 4, 6, 8]):
                mlflow.log_metric(f"val_precision_NRS_{nrs}", precision_por_clase[idx])
                mlflow.log_metric(f"val_recall_NRS_{nrs}",    recall_por_clase[idx])

            cm = confusion_matrix(val_targets, val_preds, labels=[0, 1, 2, 3])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=[2,4,6,8], yticklabels=[2,4,6,8])
            plt.title(f"Matriz de Confusión - Fold {fold+1}")
            plt.ylabel("Real (NRS)")
            plt.xlabel("Predicho (NRS)")
            cm_path = f"confusion_matrix_fold{fold+1}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

            mlflow.pytorch.log_model(
                pytorch_model=model, 
                artifact_path=f"modelo_eegnet_fold{fold+1}", 
                serialization_format="pickle"
            )

            ruta_modelo_local = f"eegnet_fold{fold+1}.pth"
            torch.save(model.state_dict(), ruta_modelo_local)
            print(f"   -> Copia local de los pesos guardada en: {os.path.abspath(ruta_modelo_local)}")

            print(f"\n   ✓ Fold {fold+1} completado exitosamente")
            print(f"      Macro F1 : {macro_f1:.4f} | Accuracy : {acc:.4f} | MCC : {mcc:.4f}\n")

    print("=" * 60)
    print("Entrenamiento finalizado exitosamente. Modelos .pth listos en tu computadora.")