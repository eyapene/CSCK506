from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt


def create_model(input_dim=784,
                 first_layer=256,
                 second_layer=128,
                 activation_func='relu',
                 learning_rate=0.05,
                 momentum=0.5,
                 use_dropout=False,
                 dropout_rate=0.2):
    """
    Build and compile a configurable feedforward neural network for multi-class classification.

    This function allows flexible experimentation with layer sizes, activation functions,
    learning parameters, and optional dropout regularization.

    Architecture:
        - Input layer: 'input_dim' features (e.g., 784 for flattened 28x28 images)
        - Hidden layer 1: 'first_layer' neurons + chosen activation
        - Hidden layer 2: 'second_layer' neurons + chosen activation
        - Output layer: 10 neurons with Softmax activation (probabilities for 10 classes)

    Args:
        input_dim (int): Number of input features.
        first_layer (int): Number of neurons in the first hidden layer.
        second_layer (int): Number of neurons in the second hidden layer.
        activation_func (str): Activation function for hidden layers ('relu', 'sigmoid', etc.).
        learning_rate (float): Learning rate for SGD optimizer.
        momentum (float): Momentum for SGD optimizer.
        use_dropout (bool): Whether to include dropout for regularization.
        dropout_rate (float): Fraction of neurons dropped if dropout is used.

    Returns:
        model (keras.Model): Compiled Keras Sequential model ready for training.
    """

    # 1. Initialize the Sequential model
    # Sequential: simple linear stack of layers
    model = Sequential()

    # 2. First hidden layer
    # Processes the input features and introduces non-linearity
    model.add(Dense(first_layer, activation=activation_func, input_shape=(input_dim,)))
    if use_dropout:
        # Dropout randomly disables neurons to reduce overfitting
        model.add(Dropout(dropout_rate))

    # 3. Second hidden layer
    # Learns higher-level feature combinations from first hidden layer
    model.add(Dense(second_layer, activation=activation_func))
    if use_dropout:
        model.add(Dropout(dropout_rate))

    # 4. Output layer
    # Produces class probabilities using softmax for multi-class classification
    model.add(Dense(10, activation='softmax'))

    # 5. Define the optimizer
    # SGD: Stochastic Gradient Descent with optional momentum to accelerate convergence
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

    # 6. Compile the model
    # Categorical Crossentropy: appropriate loss for multi-class classification
    # Accuracy: standard metric for evaluation
    model.compile(optimizer=optimizer,
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def train_baseline_model(X_train, y_train, X_test, y_test,
                         activation_func='sigmoid',
                         use_dropout=False,
                         dropout_rate=0.2,
                         epochs=10,
                         batch_size=1000,
                         verbose=1):
    """
    Build, configure, and train a neural network using the 'create_model' function.

    This function serves as a high-level wrapper to instantiate a neural network
    with flexible hyperparameters, train it on provided training data, and evaluate
    on test data. It simplifies experimentation with activation functions, dropout,
    and training settings.

    Architecture:
        - Uses 'create_model' to define the network architecture.
        - Trains using specified epochs and batch size.
        - Tracks validation performance during training.

    Args:
        X_train (array-like): Training input data (features).
        y_train (array-like): One-hot encoded training labels.
        X_test (array-like): Test input data for validation during training.
        y_test (array-like): One-hot encoded test labels.
        activation_func (str): Activation function for hidden layers ('sigmoid', 'relu', etc.).
        use_dropout (bool): Whether to include dropout layers for regularization.
        dropout_rate (float): Fraction of neurons dropped if dropout is used.
        epochs (int): Number of training iterations over the full dataset.
        batch_size (int): Number of samples per gradient update.
        verbose (int): Verbosity level of training output (0, 1, or 2).

    Returns:
        model (keras.Model): The trained Keras model ready for evaluation or prediction.
        history (keras.callbacks.History): History object containing training and validation metrics.
    """

    # 1. Create the model
    # Leverages the previously defined 'create_model' function
    # Allows easy modification of activation function and dropout behavior
    model = create_model(
        activation_func=activation_func,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate
    )

    # 2. Train the model
    # model.fit trains the network on training data and validates on test data
    # Tracks metrics (e.g., accuracy) per epoch
    history = model.fit(
        X_train, y_train,                 # Training data
        epochs=epochs,                    # Number of full passes over the dataset
        batch_size=batch_size,            # Number of samples processed before updating weights
        validation_data=(X_test, y_test),# Evaluate performance on unseen data after each epoch
        verbose=verbose                   # Control training progress display
    )

    # 3. Return trained model and training history
    # model: ready for predictions or further evaluation
    # history: contains loss and metric values across all epochs for plotting or analysis
    return model, history


def plot_model_predictions(model, images, labels, class_names, title="Model Predictions"):
    """
    Predicts labels for input images and visualizes them in a 5x5 grid.

    Correct predictions are shown in green; incorrect predictions are shown in red.
    Useful for quickly inspecting model performance on sample images.

    Args:
        model (keras.Model): Trained Keras model used for predictions.
        images (array-like): Input images (flattened or original shape) to predict.
        labels (array-like): True one-hot encoded labels corresponding to 'images'.
        class_names (list of str): Names of classes for labeling predictions.
        title (str): Title for the overall plot.

    Returns:
        None: Displays a matplotlib plot.
    """

    # 1. Generate predictions for all input images
    # Each prediction is a vector of probabilities for each class
    predictions = model.predict(images)

    # 2. Set up the figure for a 5x5 grid
    plt.figure(figsize=(12, 12))

    # 3. Loop through the first 25 images
    for i in range(25):
        plt.subplot(5, 5, i + 1)  # Position each image in the 5x5 grid
        plt.xticks([])             # Hide x-axis ticks
        plt.yticks([])             # Hide y-axis ticks
        plt.grid(False)            # Disable grid lines

        # 4. Reshape flattened images (28x28) for visualization
        plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)

        # 5. Identify the predicted and true labels
        predicted_label = np.argmax(predictions[i])  # Index of highest probability
        true_label = np.argmax(labels[i])           # Index of actual label

        # 6. Set color: Green for correct prediction, Red for incorrect
        color = 'green' if predicted_label == true_label else 'red'

        # 7. Add label below image showing predicted and true classes
        plt.xlabel(
            f"Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}",
            color=color,
            fontsize=9
        )

    # 8. Adjust layout and add overall title
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)  # y>1 moves title above plot
    plt.show()

def kfold_grid_search(X_train, y_train,
                      activation_func='relu',
                      use_dropout=False,
                      param_grid=None,
                      epochs=10,
                      batch_size=1000,
                      n_splits=5,
                      verbose=0):
    """
    Perform K-Fold cross-validation with grid search over hyperparameters for a neural network.

    This function systematically explores combinations of hyperparameters, trains a model
    for each combination using K-Fold CV, and identifies the best parameters based on
    average validation accuracy.

    Args:
        X_train (array-like): Training input data (features).
        y_train (array-like): One-hot encoded training labels.
        activation_func (str): Activation function for hidden layers ('relu', 'sigmoid', etc.).
        use_dropout (bool): Whether to include dropout in the network.
        param_grid (dict): Dictionary of hyperparameter lists to search over. Must include
            keys: 'learning_rate', 'momentum', 'first_layer', 'second_layer', 
            optionally 'dropout_rate' if use_dropout=True.
        epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        n_splits (int): Number of folds for K-Fold cross-validation.
        verbose (int): Verbosity level for model training.

    Returns:
        best_params (dict): Hyperparameter combination achieving the highest CV accuracy.
        best_score (float): Highest mean validation accuracy achieved across folds.
    """

    # 1. Validate inputs
    if param_grid is None:
        raise ValueError("param_grid must be provided.")

    if use_dropout and "dropout_rate" not in param_grid:
        raise ValueError("'dropout_rate' must be provided when use_dropout=True.")

    required_keys = ["learning_rate", "momentum", "first_layer", "second_layer"]
    for key in required_keys:
        if key not in param_grid:
            raise ValueError(f"Missing required param_grid key: {key}")

    # 2. Extract input dimension from training data
    input_dim = X_train.shape[1]

    # 3. Initialize K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = 0       # Tracks best mean CV accuracy
    best_params = None   # Tracks hyperparameters achieving best score

    # 4. Set dropout values to iterate over
    dropout_values = param_grid["dropout_rate"] if use_dropout else [0.0]

    # 5. Grid search: loop over all hyperparameter combinations
    for lr in param_grid["learning_rate"]:
        for mom in param_grid["momentum"]:
            for l1 in param_grid["first_layer"]:
                for l2 in param_grid["second_layer"]:
                    for dr in dropout_values:

                        fold_accuracies = []  # Store validation accuracies for each fold

                        # 6. K-Fold cross-validation loop
                        for train_index, val_index in kf.split(X_train):

                            # Split training data into current fold's train and validation sets
                            X_tr, X_val = X_train[train_index], X_train[val_index]
                            y_tr, y_val = y_train[train_index], y_train[val_index]

                            # 7. Create and compile model with current hyperparameters
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

                            # 8. Train model on fold's training data
                            model.fit(
                                X_tr, y_tr,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=verbose
                            )

                            # 9. Evaluate model on fold's validation data
                            _, val_acc = model.evaluate(X_val, y_val, verbose=0)
                            fold_accuracies.append(val_acc)

                        # 10. Compute mean accuracy across folds for this hyperparameter combo
                        mean_acc = np.mean(fold_accuracies)

                        # 11. Print current hyperparameter combination and CV accuracy
                        print(
                            f"LR={lr}, Mom={mom}, L1={l1}, L2={l2}"
                            f"{f', Dropout={dr}' if use_dropout else ''}"
                            f" → CV Acc={mean_acc:.4f}"
                        )

                        # 12. Update best parameters if current mean accuracy is higher
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

    # 13. Print summary of best hyperparameters and accuracy
    print("\nBest Parameters:", best_params)
    print(f"Best Cross Validation Accuracy: {best_score:.4f}")

    # 14. Return best parameters and score
    return best_params, best_score

def train_tuned_model(X_train, y_train,
                      X_test, y_test,
                      activation_func,
                      best_params,
                      use_dropout=False,
                      epochs=10,
                      batch_size=1000):
    """
    Train a neural network using the best hyperparameters obtained from grid search.

    This function validates the best parameters, creates a model using 'create_model',
    and trains it on the full training set while evaluating on test data.

    Args:
        X_train (array-like): Training input features.
        y_train (array-like): One-hot encoded training labels.
        X_test (array-like): Test input features for validation.
        y_test (array-like): One-hot encoded test labels.
        activation_func (str): Activation function for hidden layers.
        best_params (dict): Best hyperparameters from grid search.
        use_dropout (bool): Whether to include dropout.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.

    Returns:
        model (keras.Model): Trained Keras model.
        history (keras.callbacks.History): Training history object.
        test_acc (float): Final accuracy on the test set.
    """

    # 1. Validate required hyperparameters
    required_keys = ["learning_rate", "momentum", "first_layer", "second_layer"]
    for key in required_keys:
        if key not in best_params:
            raise ValueError(f"Missing required best_params key: {key}")

    if use_dropout and "dropout_rate" not in best_params:
        raise ValueError("dropout_rate missing in best_params while use_dropout=True")

    input_dim = X_train.shape[1]

    # 2. Create model using best hyperparameters
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

    # 3. Train model and validate on test data
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return model, history


def extract_metrics(history, model_name=None):
    """
    Extract key training and validation metrics from a Keras History object.

    Converts history into NumPy arrays and computes mean and standard deviation
    for training/validation accuracy and loss.

    Args:
        history (keras.callbacks.History): History object returned by model.fit.
        model_name (str, optional): Name to associate with the model metrics.

    Returns:
        dict: Dictionary containing:
            - Model name
            - Epoch-wise train/test loss and accuracy
            - Average and standard deviation for train/test accuracy
            - Average train/test loss
    """

    hist = history.history

    # 1. Convert history lists to NumPy arrays for computation
    train_loss = np.array(hist['loss'])
    test_loss = np.array(hist['val_loss'])
    train_acc = np.array(hist['accuracy'])
    test_acc = np.array(hist['val_accuracy'])

    # 2. Return a structured dictionary of metrics
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
    Plot training and validation metrics from a single model and display a summary table.

    Uses `extract_metrics` to gather epoch-wise losses and accuracies, 
    then creates a figure with:
        - Loss plot
        - Accuracy plot
        - Summary table with averages and standard deviations

    Args:
        history (keras.callbacks.History): Training history from model.fit.
        model_name (str): Name of the model for labeling plots and tables.

    Returns:
        dict: Extracted metrics from `extract_metrics` for further analysis.
    """

    # 1. Extract metrics
    metrics = extract_metrics(history, model_name)

    train_loss = metrics["train_loss"]
    test_loss = metrics["test_loss"]
    train_acc = metrics["train_acc"]
    test_acc = metrics["test_acc"]

    # -----------------------
    # 2. Create Figure Layout
    # -----------------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_loss = fig.add_subplot(gs[0, 0])   # Loss plot
    ax_acc = fig.add_subplot(gs[0, 1])    # Accuracy plot
    ax_table = fig.add_subplot(gs[1, :])  # Summary table

    # -----------------------
    # 3. Plot Loss
    # -----------------------
    ax_loss.plot(train_loss, label='Train Loss')
    ax_loss.plot(test_loss, label='Test Loss')
    ax_loss.set_title(f'{model_name} - Model Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.legend()

    # -----------------------
    # 4. Plot Accuracy
    # -----------------------
    ax_acc.plot(train_acc, label='Train Accuracy')
    ax_acc.plot(test_acc, label='Test Accuracy')
    ax_acc.set_title(f'{model_name} - Model Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()

    # -----------------------
    # 5. Summary Table
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

    # Bold first column header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.show()

    return metrics


def plot_model_comparison(results, title=""):
    """
    Plot a stakeholder-ready comparison table of multiple models.

    Converts a list of metric dictionaries into a formatted table displaying:
        - Test Accuracy (avg ± std)
        - Test Loss (avg)
        - Train Accuracy (avg ± std)
        - Train Loss (avg)

    Args:
        results (list of dicts): Each dict contains model metrics (from extract_metrics).
        title (str): Title for the table.

    Returns:
        pd.DataFrame: Formatted DataFrame used for plotting the table.
    """

    import pandas as pd

    # 1. Convert results into a DataFrame
    df = pd.DataFrame(results)

    # 2. Keep only relevant metrics
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
    # 3. Format metrics for presentation
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

    df_final = df[[
        "Model",
        "Test Accuracy (Avg)",
        "Test Loss (Avg)",
        "Avg Train Accuracy",
        "Avg Train Loss"
    ]]

    # -----------------------------
    # 4. Plot Table
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
