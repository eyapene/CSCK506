from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

def create_model(activation_func='sigmoid', use_dropout=False, rate=0.2):

    # 1. Initialize the Sequential model
    # This is a linear stack of layers where each layer has exactly one input tensor and one output tensor.
    model = Sequential()

    # 2. First Hidden Layer
    # We use 256 neurons. 'input_shape=(784,)' matches our flattened 28x28 images.
    # Sigmoid: σ(x) = 1 / (1 + exp(-x)). It squashes values between 0 and 1.
    model.add(Dense(256, activation=activation_func, input_shape=(784,)))
    if use_dropout:
        model.add(Dropout(rate))

    # 3. Second Hidden Layer
    # Another dense layer to extract higher-level combinations of the first layer's features.  We use 128 neurons
    model.add(Dense(128, activation=activation_func))
    if use_dropout:
        model.add(Dropout(rate))

    # 4. Output Layer
    # 10 neurons (one for each Fashion MNIST class).
    # Softmax turns the output into a probability distribution (sums to 1.0).
    # Softmax squashes the 10 outputs such that they all sum to 1.0. This is required for Multi-class classification (e.g., an image is either a T-shirt OR a Bag, but not both).
    model.add(Dense(10, activation='softmax'))

    # 5. Compile the Model
    # Optimizer: SGD (Stochastic Gradient Descent) updates weights based on the gradient.
    # Loss: 'categorical_crossentropy' measures the "distance" between the predicted and true label.
    optimizer = SGD(learning_rate=0.05, momentum=0.5)

    # 6. Model compilation
    model.compile(optimizer=optimizer,
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def run_kfold_experiment(X_train, y_train, activation='sigmoid', use_dropout=False):
    # Lists to store the history of each fold
    all_fold_train_acc = []
    all_fold_val_acc = []
    all_fold_train_loss = []
    all_fold_val_loss = []

    # Accuracy and Loss scores for the final summary
    acc_per_fold = []
    loss_per_fold = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_no = 1

    for train, val in kf.split(X_train, y_train):

        print(f"\nTraining Fold {fold_no} for "
              f"{activation}{' + dropout' if use_dropout else ''} Model...")

        # Create model
        model = create_model(activation_func=activation, use_dropout=use_dropout)

        # Train the model
        history = model.fit(
            X_train[train], y_train[train],
            shuffle=False,
            epochs=10,
            batch_size=1000,
            validation_data=(X_train[val], y_train[val]),
            verbose=0
        )

        # Store full history (for plotting later)
        all_fold_train_acc.append(history.history['accuracy'])
        all_fold_val_acc.append(history.history['val_accuracy'])
        all_fold_train_loss.append(history.history['loss'])
        all_fold_val_loss.append(history.history['val_loss'])

        # Get final epoch metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        # Store validation metrics for summary
        acc_per_fold.append(final_val_acc * 100)
        loss_per_fold.append(final_val_loss)

        # Print fold results
        """
        print(f"Fold {fold_no} Results:")
        print(f"  Train Loss: {final_train_loss:.4f}")
        print(f"  Train Accuracy: {final_train_acc*100:.2f}%")
        print(f"  Validation Loss: {final_val_loss:.4f}")
        print(f"  Validation Accuracy: {final_val_acc*100:.2f}%")
        """
        print(f'Train Loss: {final_train_loss:.4f} - Train Accuracy: {final_train_acc*100:.2f}% - Val Loss: {final_val_loss:.4f} - Val Accuracy: {final_val_acc*100:.2f}%')

        fold_no += 1

    # Print final K-Fold summary
    print("\n==== K-FOLD CROSS VALIDATION RESULTS ====")
    print(f"Average Validation Accuracy: {np.mean(acc_per_fold):.2f}% "
          f"(+/- {np.std(acc_per_fold):.2f})")
    print(f"Average Validation Loss: {np.mean(loss_per_fold):.4f}")

    return {
        "train_acc": all_fold_train_acc,
        "val_acc": all_fold_val_acc,
        "train_loss": all_fold_train_loss,
        "val_loss": all_fold_val_loss,
        "avg_acc": np.mean(acc_per_fold),
        "avg_loss": np.mean(loss_per_fold),
        "std_acc": np.std(acc_per_fold)
    }

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

# Execute plotting
#plot_kfold_history(all_fold_train_acc, all_fold_val_acc, 'Accuracy', 'Sigmoid MLP')
#plot_kfold_history(all_fold_train_loss, all_fold_val_loss, 'Loss', 'Sigmoid MLP')


def train_test_model(X_train, y_train, X_test, y_test,
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
        shuffle=False,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=verbose
    )

    return model, history


import numpy as np
import matplotlib.pyplot as plt

def plot_final_metrics(history):
    """
    Plots training history side-by-side and adds a summary table
    inside the figure.
    """

    hist = history.history

    train_loss = np.array(hist['loss'])
    test_loss = np.array(hist['val_loss'])
    train_acc = np.array(hist['accuracy'])
    test_acc = np.array(hist['val_accuracy'])

    # -----------------------
    # Compute Summary Stats
    # -----------------------
    summary_data = [
        ["Train Accuracy (avg ± std)",
         f"{train_acc.mean()*100:.2f}% ± {train_acc.std()*100:.2f}"],
        ["Test Accuracy (avg ± std)",
         f"{test_acc.mean()*100:.2f}% ± {test_acc.std()*100:.2f}"],
        ["Train Loss (avg)",
         f"{train_loss.mean():.4f}"],
        ["Test Loss (avg)",
         f"{test_loss.mean():.4f}"]
    ]

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
    ax_loss.set_title('Model Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.legend()

    # -----------------------
    # Accuracy Plot
    # -----------------------
    ax_acc.plot(train_acc, label='Train Accuracy')
    ax_acc.plot(test_acc, label='Test Accuracy')
    ax_acc.set_title('Model Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

    # -----------------------
    # Add Table
    # -----------------------
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

    # Make header bold
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.show()

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

    # Force user to provide param_grid
    if param_grid is None:
        print("Warning: You must provide a param_grid dictionary.")
        print("Example:")
        print("""
            param_grid = {
                "learning_rate": [0.03, 0.07],
                "momentum": [0.8, 1.0],
                "layer1_units": [512],
                "layer2_units": [256],
                "dropout_rate": [0.1, 0.3]  # Only if use_dropout=True
            }
        """)
        return None, None

    # Validate dropout configuration
    if use_dropout and "dropout_rate" not in param_grid:
        print("Warning: 'dropout_rate' must be provided in param_grid when use_dropout=True.")
        return None, None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = 0
    best_params = None

    dropout_values = param_grid["dropout_rate"] if use_dropout else [None]

    for lr in param_grid["learning_rate"]:
        for mom in param_grid["momentum"]:
            for l1 in param_grid["layer1_units"]:
                for l2 in param_grid["layer2_units"]:
                    for dr in dropout_values:

                        fold_accuracies = []

                        for train_index, val_index in kf.split(X_train):

                            X_tr, X_val = X_train[train_index], X_train[val_index]
                            y_tr, y_val = y_train[train_index], y_train[val_index]

                            model = Sequential()
                            model.add(Dense(l1, activation=activation_func, input_shape=(784,)))

                            if use_dropout:
                                model.add(Dropout(dr))

                            model.add(Dense(l2, activation=activation_func))

                            if use_dropout:
                                model.add(Dropout(dr))

                            model.add(Dense(10, activation='softmax'))

                            optimizer = SGD(learning_rate=lr, momentum=mom)

                            model.compile(
                                optimizer=optimizer,
                                loss=CategoricalCrossentropy(),
                                metrics=['accuracy']
                            )

                            model.fit(
                                X_tr, y_tr,
                                shuffle=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=verbose
                            )

                            _, val_acc = model.evaluate(X_val, y_val, verbose=0)
                            fold_accuracies.append(val_acc)

                        mean_acc = np.mean(fold_accuracies)

                        print(
                            f"LR={lr}, MOM={mom}, L1={l1}, L2={l2}"
                            f"{f', DR={dr}' if use_dropout else ''} "
                            f"→ CV Acc={mean_acc:.4f}"
                        )

                        if mean_acc > best_score:
                            best_score = mean_acc
                            best_params = {
                                "learning_rate": lr,
                                "momentum": mom,
                                "layer1_units": l1,
                                "layer2_units": l2
                            }

                            if use_dropout:
                                best_params["dropout_rate"] = dr

    print("\nBest Parameters:", best_params)
    print("Best CV Accuracy:", best_score)

    return best_params, best_score

def train_final_model(X_train, y_train, X_test, y_test,
                      activation_func,
                      best_params,
                      use_dropout=False,
                      epochs=10,
                      batch_size=1000):

    model = Sequential()
    model.add(Dense(best_params["layer1_units"],
                    activation=activation_func,
                    input_shape=(784,)))

    if use_dropout:
        model.add(Dropout(best_params["dropout_rate"]))

    model.add(Dense(best_params["layer2_units"],
                    activation=activation_func))

    if use_dropout:
        model.add(Dropout(best_params["dropout_rate"]))

    model.add(Dense(10, activation='softmax'))

    optimizer = SGD(
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"]
    )

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        shuffle=False,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return model, history
