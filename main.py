import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data
from model import MLP, SurrogateMLP
from train import train_model, train_surrogate, adversarial_training
from attacks import (
    fgsm_attack,
    pgd_attack,
    bim_attack,
    transfer_attack,
    random_attack,
    square_attack,
    fgsm_attack_def,
)
from evaluation import evaluate, evaluate_attack, evaluate_f1, plot_confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler("log_or_text_variant.txt", mode="w", encoding="utf-8")
                    ])
logger = logging.getLogger()

def log_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    logger.info(message)
# ==============================
# ===== 1. ЗАГРУЗКА ДАННЫХ =====
# ==============================

(train_loader, test_loader,
 X_train_tensor, y_train_tensor,
 X_test_tensor, y_test_tensor,
 label_encoder, input_dim, num_classes) = load_data("train.csv", "test.csv")

# ==================================
# ===== 2. ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ =====
# ==================================
model = MLP(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =======================================
# ===== 3. ОБУЧЕНИЕ ОСНОВНОЙ МОДЕЛИ =====
# =======================================

num_epochs = 100
train_model(model, train_loader, criterion, optimizer, num_epochs)
acc_before_attack = evaluate(model, test_loader)
log_print(f"Accuracy before attack: {acc_before_attack * 100:.2f}%")

# ==============================
# ===== 4. WHITE-BOX АТАКИ =====
# ==============================

X_test_adv_fgsm = fgsm_attack(model, X_test_tensor, y_test_tensor, epsilon=0.1)
X_test_adv_pgd = pgd_attack(model, X_test_tensor, y_test_tensor, epsilon=0.1, alpha=0.01, num_iter=40)
X_test_adv_bim = bim_attack(model, X_test_tensor, y_test_tensor, epsilon=0.3, alpha=0.005, num_iter=10)

acc_fgsm = evaluate_attack(model, X_test_adv_fgsm, y_test_tensor)
acc_pgd = evaluate_attack(model, X_test_adv_pgd, y_test_tensor)
acc_bim = evaluate_attack(model, X_test_adv_bim, y_test_tensor)
log_print(f"White-box FGSM attack accuracy: {acc_fgsm * 100:.2f}%")
log_print(f"White-box PGD attack accuracy: {acc_pgd * 100:.2f}%")
log_print(f"White-box BIM attack accuracy: {acc_bim * 100:.2f}%")

# ==============================
# ===== 5. BLACK-BOX АТАКИ =====
# ==============================

surrogate_model = SurrogateMLP(input_dim, num_classes)
criterion_sur = nn.CrossEntropyLoss()
optimizer_sur = optim.Adam(surrogate_model.parameters(), lr=0.001)
num_epochs_sur = 10
train_surrogate(surrogate_model, train_loader, criterion_sur, optimizer_sur, num_epochs_sur)

X_test_adv_transfer = transfer_attack(surrogate_model, X_test_tensor, y_test_tensor, epsilon=0.1)
X_test_adv_random = random_attack(model, X_test_tensor, y_test_tensor, epsilon=5, num_trials=200)
X_test_adv_square = square_attack(model, X_test_tensor, y_test_tensor, epsilon=0.5)

acc_transfer = evaluate_attack(model, X_test_adv_transfer, y_test_tensor)
acc_random = evaluate_attack(model, X_test_adv_random, y_test_tensor)
acc_square = evaluate_attack(model, X_test_adv_square, y_test_tensor)
log_print(f"Black-box Transfer attack accuracy: {acc_transfer * 100:.2f}%")
log_print(f"Black-box Random attack accuracy: {acc_random * 100:.2f}%")
log_print(f"Accuracy after Square attack: {acc_square * 100:.2f}%")

# =======================================================
# ===== 6. Защищённая модель (Adversarial Training) =====
# =======================================================
model_def = MLP(input_dim, num_classes)
optimizer_def = optim.Adam(model_def.parameters(), lr=0.001)
num_epochs_def = 100
adversarial_training(model_def, train_loader, criterion, optimizer_def, num_epochs_def, fgsm_attack_def)

acc_def_before_attack = evaluate(model_def, test_loader)
log_print(f"Defended Model Accuracy before attack: {acc_def_before_attack * 100:.2f}%")

X_test_adv_fgsm_def = fgsm_attack(model_def, X_test_tensor, y_test_tensor, epsilon=0.1)
X_test_adv_pgd_def  = pgd_attack(model_def, X_test_tensor, y_test_tensor, epsilon=0.1, alpha=0.01, num_iter=40)
X_test_adv_bim_def  = bim_attack(model_def, X_test_tensor, y_test_tensor, epsilon=0.1, alpha=0.005, num_iter=10)

acc_fgsm_def = evaluate_attack(model_def, X_test_adv_fgsm_def, y_test_tensor)
acc_pgd_def  = evaluate_attack(model_def, X_test_adv_pgd_def, y_test_tensor)
acc_bim_def  = evaluate_attack(model_def, X_test_adv_bim_def, y_test_tensor)
log_print(f"Defended Model - White-box FGSM attack accuracy: {acc_fgsm_def * 100:.2f}%")
log_print(f"Defended Model - White-box PGD attack accuracy: {acc_pgd_def * 100:.2f}%")
log_print(f"Defended Model - White-box BIM attack accuracy: {acc_bim_def * 100:.2f}%")

X_test_adv_transfer_def = transfer_attack(surrogate_model, X_test_tensor, y_test_tensor, epsilon=0.1)
X_test_adv_random_def   = random_attack(model_def, X_test_tensor, y_test_tensor, epsilon=0.1, num_trials=200)
X_test_adv_square_def   = square_attack(model_def, X_test_tensor, y_test_tensor)

acc_transfer_def = evaluate_attack(model_def, X_test_adv_transfer_def, y_test_tensor)
acc_random_def   = evaluate_attack(model_def, X_test_adv_random_def, y_test_tensor)
acc_square_def   = evaluate_attack(model_def, X_test_adv_square_def, y_test_tensor)
log_print(f"Defended Model - Black-box Transfer attack accuracy: {acc_transfer_def * 100:.2f}%")
log_print(f"Defended Model - Black-box Random attack accuracy: {acc_random_def * 100:.2f}%")
log_print(f"Defended Model - Black-box Square attack accuracy: {acc_square_def * 100:.2f}%")

# =======================================================
# ===== 7. Анализ результатов и сохранение графиков =====
# =======================================================

log_print("\n--- Оценка до атак ---")
acc_before_attack = evaluate(model, test_loader)
f1_before_attack = evaluate_f1(model, test_loader)
log_print(f"Accuracy before attack: {acc_before_attack * 100:.2f}%")
log_print(f"F1-score before attack: {f1_before_attack:.4f}")

log_print("\n--- Оценка после White-box атак ---")
f1_fgsm = evaluate_f1(model, DataLoader(TensorDataset(X_test_adv_fgsm, y_test_tensor), batch_size=32))
f1_pgd = evaluate_f1(model, DataLoader(TensorDataset(X_test_adv_pgd, y_test_tensor), batch_size=32))
f1_bim = evaluate_f1(model, DataLoader(TensorDataset(X_test_adv_bim, y_test_tensor), batch_size=32))
log_print(f"White-box FGSM attack accuracy: {acc_fgsm * 100:.2f}% | F1-score: {f1_fgsm:.4f}")
log_print(f"White-box PGD attack accuracy: {acc_pgd * 100:.2f}% | F1-score: {f1_pgd:.4f}")
log_print(f"White-box BIM attack accuracy: {acc_bim * 100:.2f}% | F1-score: {f1_bim:.4f}")

log_print("\n--- Оценка после Black-box атак ---")
f1_transfer = evaluate_f1(model, DataLoader(TensorDataset(X_test_adv_transfer, y_test_tensor), batch_size=32))
f1_random = evaluate_f1(model, DataLoader(TensorDataset(X_test_adv_random, y_test_tensor), batch_size=32))
f1_square = evaluate_f1(model, DataLoader(TensorDataset(X_test_adv_square, y_test_tensor), batch_size=32))
log_print(f"Black-box Transfer attack accuracy: {acc_transfer * 100:.2f}% | F1-score: {f1_transfer:.4f}")
log_print(f"Black-box Random attack accuracy: {acc_random * 100:.2f}% | F1-score: {f1_random:.4f}")
log_print(f"Accuracy after Square attack: {acc_square * 100:.2f}% | F1-score: {f1_square:.4f}")

log_print("\n--- Оценка защищённой модели до атак ---")
acc_def_before_attack = evaluate(model_def, test_loader)
f1_def_before_attack = evaluate_f1(model_def, test_loader)
log_print(f"Defended Model Accuracy before attack: {acc_def_before_attack * 100:.2f}%")
log_print(f"Defended Model F1-score before attack: {f1_def_before_attack:.4f}")

log_print("\n--- Оценка защищённой модели после White-box атак ---")
f1_fgsm_def = evaluate_f1(model_def, DataLoader(TensorDataset(X_test_adv_fgsm_def, y_test_tensor), batch_size=32))
f1_pgd_def = evaluate_f1(model_def, DataLoader(TensorDataset(X_test_adv_pgd_def, y_test_tensor), batch_size=32))
log_print(f"Defended Model White-box FGSM attack accuracy: {acc_fgsm_def * 100:.2f}% | F1-score: {f1_fgsm_def:.4f}")
log_print(f"Defended Model White-box PGD attack accuracy: {acc_pgd_def * 100:.2f}% | F1-score: {f1_pgd_def:.4f}")

log_print("\n--- Оценка защищённой модели после Black-box атак ---")
f1_transfer_def = evaluate_f1(model_def, DataLoader(TensorDataset(X_test_adv_transfer, y_test_tensor), batch_size=32))
f1_random_def = evaluate_f1(model_def, DataLoader(TensorDataset(X_test_adv_random, y_test_tensor), batch_size=32))
f1_square_def = evaluate_f1(model_def, DataLoader(TensorDataset(X_test_adv_square, y_test_tensor), batch_size=32))
log_print(f"Defended Model Black-box Transfer attack accuracy: {acc_transfer_def * 100:.2f}% | F1-score: {f1_transfer_def:.4f}")
log_print(f"Defended Model Black-box Random attack accuracy: {acc_random_def * 100:.2f}% | F1-score: {f1_random_def:.4f}")
log_print(f"Defended Model Accuracy after Square attack: {acc_square_def * 100:.2f}% | F1-score: {f1_square_def:.4f}")

log_print("\n--- Сравнение результатов ---")
log_print(f"Accuracy Before Attack: {acc_before_attack * 100:.2f}% vs After White-box FGSM: {acc_fgsm * 100:.2f}%")
log_print(f"F1-score Before Attack: {f1_before_attack:.4f} vs After White-box FGSM: {f1_fgsm:.4f}")
log_print(f"Defended Model Accuracy Before Attack: {acc_def_before_attack * 100:.2f}% vs After White-box FGSM: {acc_fgsm_def * 100:.2f}%")

plot_confusion_matrix(model, X_test_tensor, y_test_tensor.numpy(), "Оригинальная модель (До атак)", save_path="cm_original.png")
plot_confusion_matrix(model_def, X_test_tensor, y_test_tensor.numpy(), "Защищенная модель (До атак)", save_path="cm_defended_before.png")
plot_confusion_matrix(model, X_test_adv_fgsm, y_test_tensor.numpy(), "White-box FGSM Attack", save_path="cm_fgsm.png")
plot_confusion_matrix(model_def, X_test_adv_fgsm_def, y_test_tensor.numpy(), "Defended Model (FGSM Attack)", save_path="cm_defended_fgsm.png")