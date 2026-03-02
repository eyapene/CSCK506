from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

def create_model(input_dim,
                 first_layer=256,
                 second_layer=128,
                 activation_func='relu',
                 learning_rate=0.01,
                 momentum=0.5,
                 use_dropout=False,
                 dropout_rate=0.2):

    # 1. Initialize the Sequential model
    # This is a linear stack of layers where each layer has exactly one input tensor and one output tensor.
    model = Sequential()

    # 2. First Hidden Layer
    # Default: 256 neurons. 'input_shape=(784,)' matches our flattened 28x28 images.
    # Sigmoid: σ(x) = 1 / (1 + exp(-x)). It squashes values between 0 and 1.
    model.add(Dense(first_layer,
                    activation=activation_func,
                    input_shape=(input_dim,)))

    if use_dropout:
        model.add(Dropout(dropout_rate))

    # 3. Second Hidden Layer
    # Default: 128 neurons
    # Another dense layer to extract higher-level combinations of the first layer's features.
    model.add(Dense(second_layer,
                    activation=activation_func))

    if use_dropout:
        model.add(Dropout(dropout_rate))

    # 4. Output Layer
    # 10 neurons (one for each Fashion MNIST class).
    # Softmax turns the output into a probability distribution (sums to 1.0).
    # Softmax squashes the 10 outputs such that they all sum to 1.0. This is required for Multi-class classification (e.g., an image is either a T-shirt OR a Bag, but not both).
    model.add(Dense(10, activation='softmax'))

    # 5. Compile the Model
    # Optimizer: SGD (Stochastic Gradient Descent) updates weights based on the gradient.
    # Loss: 'categorical_crossentropy' measures the "distance" between the predicted and true label.
    #Default: learning_rate: 0.05, Momentum: 0.5
    optimizer = SGD(learning_rate=learning_rate,
                    momentum=momentum)

    # 6. Model compilation
    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def plot_kfold_history(train_histories, val_histories, metric_name, model_name):
    # Convert lists to numpy arrays for calculation: Shape (Folds, Epochs)
    train_arr = np.array(train_histories)
    val_arr = np.array(val_histories)

    # Calculate Mean and Standard Deviation
    train_mean = np.mean(train_arr, axis=0)
    train_std = np.std(train_arr, axis=0)
    val_mean = np.mean(val_arr, axis=0)
    val_std = np.std(val_arr, axis=0)

    epochs = range(1, len(train_mean) + 1)

    plt.figure(figsize=(8, 5))

    # Plot Mean Lines
    plt.plot(epochs, train_mean, label=f'Mean Train {metric_name}', color='blue', lw=2)
    plt.plot(epochs, val_mean, label=f'Mean Test {metric_name}', color='orange', lw=2)

    # Add Shaded Area for Variance (Standard Deviation)
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.1)

    plt.title(f'K-Fold Cross-Validation: {model_name} {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()


def train_baseline_model(X_train, y_train, X_test, y_test,
                activation_func='sigmoid',
                use_dropout=False,
                dropout_rate=0.2,
                epochs=10,
                batch_size=1000,
                verbose=1):
    """
    Creates and trains a neural network using the specified configuration.

    Parameters:
    -----------
    activation_func : str
        Activation function for hidden layers ('sigmoid', 'relu', etc.)
    use_dropout : bool
        Whether to include dropout layers
    rate : float
        Dropout rate (if enabled)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training

    Returns:
    --------
    model : trained Keras model
    history : training history object
    """

    # Create model with full flexibility
    model = create_model(
        activation_func=activation_func,
        use_dropout=use_dropout,
        rate=dropout_rate
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        #validation_data=(X_test, y_test),
        verbose=verbose
    )
    # Final evaluation on test set (AFTER training)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    return model, history

def plot_model_predictions(model, images, labels, class_names, title="Model Predictions"):
    """
    Predicts labels for the provided images and plots a 5x5 grid of results.
    Correct predictions are labeled in green; incorrect ones in red.
    """
    # 1. Generate predictions for the input images
    predictions = model.predict(images)

    plt.figure(figsize=(12, 12))

    # 2. Iterate through the first 25 images
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # Reshape the flattened 784 vector back to 28x28 for display
        plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)

        # Determine predicted and true labels
        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(labels[i])

        # Define color based on accuracy: Green if correct, Red if wrong
        color = 'green' if predicted_label == true_label else 'red'

        plt.xlabel(f"Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}",
                   color=color, fontsize=9)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()

def kfold_grid_search(X_train, y_train,
                      activation_func='relu',
                      use_dropout=False,
                      param_grid=None,
                      epochs=10,
                      batch_size=1000,
                      n_splits=5,
                      verbose=0):

    if param_grid is None:
        raise ValueError("param_grid must be provided.")

    if use_dropout and "dropout_rate" not in param_grid:
        raise ValueError("'dropout_rate' must be provided when use_dropout=True.")

    required_keys = ["learning_rate", "momentum", "first_layer", "second_layer"]
    for key in required_keys:
        if key not in param_grid:
            raise ValueError(f"Missing required param_grid key: {key}")

    input_dim = X_train.shape[1]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = 0
    best_params = None

    dropout_values = param_grid["dropout_rate"] if use_dropout else [0.0]

    for lr in param_grid["learning_rate"]:
        for mom in param_grid["momentum"]:
            for l1 in param_grid["first_layer"]:
                for l2 in param_grid["second_layer"]:
                    for dr in dropout_values:

                        fold_accuracies = []

                        for train_index, val_index in kf.split(X_train):

                            X_tr, X_val = X_train[train_index], X_train[val_index]
                            y_tr, y_val = y_train[train_index], y_train[val_index]

                            model = create_model(
                                input_dim=input_dim,
                                first_layer=l1,
                                second_layer=l2,
                                activation_func=activation_func,
                                learning_rate=lr,
                                momentum=mom,
                                use_dropout=use_dropout,
                                dropout_rate=dr
                            )

                            model.fit(
                                X_tr, y_tr,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=verbose
                            )

                            _, val_acc = model.evaluate(X_val, y_val, verbose=0)
                            fold_accuracies.append(val_acc)

                        mean_acc = np.mean(fold_accuracies)

                        print(
                            f"LR={lr}, Mom={mom}, L1={l1}, L2={l2}"
                            f"{f', Dropout={dr}' if use_dropout else ''}"
                            f" → CV Acc={mean_acc:.4f}"
                        )

                        if mean_acc > best_score:
                            best_score = mean_acc
                            best_params = {
                                "learning_rate": lr,
                                "momentum": mom,
                                "first_layer": l1,
                                "second_layer": l2
                            }

                            if use_dropout:
                                best_params["dropout_rate"] = dr

    print("\nBest Parameters:", best_params)
    print(f"Best Cross Validation Accuracy: {best_score:.4f}")

    return best_params, best_score

def train_final_model(X_train, y_train,
                      X_test, y_test,
                      activation_func,
                      best_params,
                      use_dropout=False,
                      epochs=10,
                      batch_size=1000):

    # Validate required parameters
    required_keys = ["learning_rate", "momentum",
                     "first_layer", "second_layer"]

    for key in required_keys:
        if key not in best_params:
            raise ValueError(f"Missing required best_params key: {key}")

    if use_dropout and "dropout_rate" not in best_params:
        raise ValueError("dropout_rate missing in best_params while use_dropout=True")

    input_dim = X_train.shape[1]

    # Reuse create_model
    model = create_model(
        input_dim=input_dim,
        first_layer=best_params["first_layer"],
        second_layer=best_params["second_layer"],
        activation_func=activation_func,
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        use_dropout=use_dropout,
        dropout_rate=best_params.get("dropout_rate", 0.0)
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Final evaluation on test set (AFTER training)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    return model, history, test_acc


def extract_metrics(history, model_name=None):
    hist = history.history

    train_loss = np.array(hist['loss'])
    test_loss = np.array(hist['val_loss'])
    train_acc = np.array(hist['accuracy'])
    test_acc = np.array(hist['val_accuracy'])

    return {
        "Model": model_name,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_acc_avg": train_acc.mean() * 100,
        "train_acc_std": train_acc.std() * 100,
        "test_acc_avg": test_acc.mean() * 100,
        "test_acc_std": test_acc.std() * 100,
        "train_loss_avg": train_loss.mean(),
        "test_loss_avg": test_loss.mean(),
    }

def plot_final_metrics(history, model_name="Model"):
    """
    Plots training history and uses extract_metrics()
    to generate the summary table.
    """

    metrics = extract_metrics(history, model_name)

    train_loss = metrics["train_loss"]
    test_loss = metrics["test_loss"]
    train_acc = metrics["train_acc"]
    test_acc = metrics["test_acc"]

    # -----------------------
    # Create Figure Layout
    # -----------------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    # -----------------------
    # Loss Plot
    # -----------------------
    ax_loss.plot(train_loss, label='Train Loss')
    ax_loss.plot(test_loss, label='Test Loss')
    ax_loss.set_title(f'{model_name} - Model Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.legend()

    # -----------------------
    # Accuracy Plot
    # -----------------------
    ax_acc.plot(train_acc, label='Train Accuracy')
    ax_acc.plot(test_acc, label='Test Accuracy')
    ax_acc.set_title(f'{model_name} - Model Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

    # -----------------------
    # Summary Table
    # -----------------------
    summary_data = [
        ["Train Accuracy (avg ± std)",
         f"{metrics['train_acc_avg']:.2f}% ± {metrics['train_acc_std']:.2f}%"],
        ["Test Accuracy (avg ± std)",
         f"{metrics['test_acc_avg']:.2f}% ± {metrics['test_acc_std']:.2f}%"],
        ["Train Loss (avg)",
         f"{metrics['train_loss_avg']:.4f}"],
        ["Test Loss (avg)",
         f"{metrics['test_loss_avg']:.4f}"]
    ]

    ax_table.axis('off')

    table = ax_table.table(
        cellText=summary_data,
        colLabels=["Metric", "Value"],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.show()

    return metrics

def plot_model_comparison(results, title=""):
    """
    Clean stakeholder-ready summary table.
    No ranking. No delta columns.
    Only formatted averages.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Keep only needed metrics
    df = df[[
        "Model",
        "test_acc_avg",
        "test_acc_std",
        "test_loss_avg",
        "train_acc_avg",
        "train_acc_std",
        "train_loss_avg"
    ]].copy()

    # -----------------------------
    # Format for presentation
    # -----------------------------
    df["Test Accuracy (Avg)"] = (
        df["test_acc_avg"].round(2).astype(str)
        + "% ± "
        + df["test_acc_std"].round(2).astype(str)
    )

    df["Test Loss (Avg)"] = df["test_loss_avg"].round(4)

    df["Avg Train Accuracy"] = (
        df["train_acc_avg"].round(2).astype(str)
        + "% ± "
        + df["train_acc_std"].round(2).astype(str)
    )

    df["Avg Train Loss"] = df["train_loss_avg"].round(4)

    # Final layout
    df_final = df[[
        "Model",
        "Test Accuracy (Avg)",
        "Test Loss (Avg)",
        "Avg Train Accuracy",
        "Avg Train Loss"
    ]]

    # -----------------------------
    # Plot Table
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    table = ax.table(
        cellText=df_final.values,
        colLabels=df_final.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    plt.title(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

    return df_final 
