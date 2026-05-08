# LightGBM Fortran API

Fortran bindings for [LightGBM](https://github.com/microsoft/LightGBM) gradient boosting library using ISO_C_BINDING.

## Features

- Direct C API bindings via `lightgbm_api` module
- High-level wrapper module `lightgbm_wrapper` for easier usage
- Support for both single and double precision data
- Binary classification and regression
- Model save/load functionality
- K-fold cross-validation
- Feature importance extraction

## Requirements

- Fortran 2003+ compiler with ISO_C_BINDING support
  - Intel Fortran (ifx/ifort)
  - GNU Fortran (gfortran)
- LightGBM library (>= 3.0)

## Installation

### 1. Install LightGBM

```bash
# Ubuntu/Debian
sudo apt-get install liblightgbm-dev

# macOS with Homebrew
brew install lightgbm

# From source
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

### 2. Build Fortran API

```bash
# Using Intel Fortran
make FC=ifx LGBM_DIR=/usr/local

# Using GNU Fortran
make FC=gfortran LGBM_DIR=/usr/local
```

## Quick Start

### Low-Level API Example

```fortran
program lgbm_simple
    use lightgbm_api
    use iso_c_binding
    implicit none

    real(c_double), target :: X(100, 5), y(100)
    real(c_double), allocatable :: predictions(:)
    type(lgbm_dataset_handle) :: dataset
    type(lgbm_booster_handle) :: booster
    integer :: i, ret
    logical :: finished

    ! Generate sample data
    call random_number(X)
    y = X(:, 1) + X(:, 2) * 0.5

    ! Create dataset
    dataset = lgbm_dataset_create_from_mat_f64(X, "")
    ret = lgbm_dataset_set_field(dataset, "label", c_loc(y), C_API_DTYPE_FLOAT64)

    ! Create and train booster
    booster = lgbm_booster_create(dataset, &
        "objective=regression metric=rmse num_leaves=31 learning_rate=0.1")

    do i = 1, 100
        finished = lgbm_booster_update_one_iter(booster)
        if (finished) exit
    end do

    ! Predict
    call lgbm_booster_predict_for_mat_f64(booster, X, predictions)

    ! Clean up
    call lgbm_booster_free(booster)
    call lgbm_dataset_free(dataset)
end program
```

### High-Level Wrapper Example

```fortran
program lgbm_wrapper_demo
    use lightgbm_wrapper
    use iso_c_binding
    implicit none

    real(c_double) :: X(1000, 10), y(1000)
    real(c_double), allocatable :: predictions(:)
    type(lgbm_model) :: model
    type(lgbm_params) :: params

    ! Generate data
    call random_number(X)
    y = X(:, 1) + X(:, 2)**2

    ! Set parameters
    params%objective = "regression"
    params%metric = "rmse"
    params%num_leaves = 31
    params%learning_rate = 0.05d0
    params%num_iterations = 100

    ! Train model
    model = lgbm_train(X(1:800, :), y(1:800), &
                       X(801:1000, :), y(801:1000), &
                       params)

    ! Predict
    call model%predict(X(801:1000, :), predictions)

    ! Save model
    call model%save("model.txt")

    ! Clean up (automatic via finalizer)
end program
```

## API Reference

### lightgbm_api Module

#### Types

- `lgbm_dataset_handle` - Opaque handle for LightGBM dataset
- `lgbm_booster_handle` - Opaque handle for LightGBM booster

#### Dataset Functions

| Function | Description |
|----------|-------------|
| `lgbm_dataset_create_from_mat_f64(data, params, reference)` | Create dataset from double precision matrix |
| `lgbm_dataset_create_from_mat_f32(data, params, reference)` | Create dataset from single precision matrix |
| `lgbm_dataset_set_field(dataset, field_name, data, dtype)` | Set label/weight/group |
| `lgbm_dataset_get_num_data(dataset)` | Get number of samples |
| `lgbm_dataset_get_num_feature(dataset)` | Get number of features |
| `lgbm_dataset_free(dataset)` | Free dataset memory |

#### Booster Functions

| Function | Description |
|----------|-------------|
| `lgbm_booster_create(train_data, params)` | Create booster for training |
| `lgbm_booster_create_from_model_file(filename, num_iter)` | Load model from file |
| `lgbm_booster_add_valid_data(booster, valid_data)` | Add validation dataset |
| `lgbm_booster_update_one_iter(booster)` | Train one iteration |
| `lgbm_booster_get_eval(booster, data_idx, results)` | Get evaluation metric |
| `lgbm_booster_predict_for_mat_f64(booster, data, predictions)` | Predict (double) |
| `lgbm_booster_predict_for_mat_f32(booster, data, predictions)` | Predict (float) |
| `lgbm_booster_save_model(booster, filename)` | Save model to file |
| `lgbm_booster_feature_importance(booster, importance)` | Get feature importance |
| `lgbm_booster_free(booster)` | Free booster memory |

#### Constants

```fortran
! Data types
C_API_DTYPE_FLOAT32 = 0
C_API_DTYPE_FLOAT64 = 1
C_API_DTYPE_INT32   = 2
C_API_DTYPE_INT64   = 3

! Prediction types
C_API_PREDICT_NORMAL     = 0
C_API_PREDICT_RAW_SCORE  = 1
C_API_PREDICT_LEAF_INDEX = 2
C_API_PREDICT_CONTRIB    = 3

! Feature importance types
C_API_FEATURE_IMPORTANCE_SPLIT = 0
C_API_FEATURE_IMPORTANCE_GAIN  = 1
```

### lightgbm_wrapper Module

#### Types

- `lgbm_model` - High-level model with automatic resource management
- `lgbm_params` - Training parameter container

#### lgbm_params Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `objective` | character(32) | "regression" | Learning objective |
| `metric` | character(64) | "rmse" | Evaluation metric |
| `num_leaves` | integer | 31 | Max leaves per tree |
| `max_depth` | integer | -1 | Max tree depth (-1=unlimited) |
| `learning_rate` | real(8) | 0.1 | Learning rate |
| `feature_fraction` | real(8) | 1.0 | Feature subsampling ratio |
| `bagging_fraction` | real(8) | 1.0 | Data subsampling ratio |
| `bagging_freq` | integer | 0 | Bagging frequency |
| `lambda_l1` | real(8) | 0.0 | L1 regularization |
| `lambda_l2` | real(8) | 0.0 | L2 regularization |
| `min_data_in_leaf` | integer | 20 | Min samples per leaf |
| `num_iterations` | integer | 100 | Number of boosting rounds |
| `verbose` | integer | 1 | Verbosity level |
| `num_threads` | integer | 0 | Threads (0=auto) |
| `seed` | integer | 0 | Random seed |

#### Wrapper Functions

| Function | Description |
|----------|-------------|
| `lgbm_train(X, y, X_valid, y_valid, params)` | Train model |
| `lgbm_predict(model, X, predictions)` | Make predictions |
| `lgbm_load_model(filename)` | Load saved model |
| `lgbm_cross_validate(X, y, params, n_folds, scores)` | K-fold CV |

## Objectives and Metrics

### Objectives

- `regression` - Regression (L2 loss)
- `regression_l1` - Regression (L1 loss)
- `binary` - Binary classification
- `multiclass` - Multi-class classification
- `lambdarank` - Learning to rank

### Metrics

- `rmse` - Root mean squared error
- `mae` - Mean absolute error
- `binary_logloss` - Binary log loss
- `auc` - Area under ROC curve
- `multi_logloss` - Multi-class log loss

## Tips for Intel Fortran (ifx)

When using Intel Fortran compiler:

```bash
# Compile with optimization
ifx -O3 -xHost lightgbm_api.f90 your_program.f90 -o program -L/path/to/lib -llightgbm

# Debug build
ifx -g -check all -traceback lightgbm_api.f90 your_program.f90 -o program_debug -L/path/to/lib -llightgbm
```

## Memory Management

The low-level API requires manual memory management:

```fortran
! Always free resources when done
call lgbm_booster_free(booster)
call lgbm_dataset_free(dataset)
```

The high-level wrapper uses Fortran finalizers for automatic cleanup.

## Troubleshooting

### Library not found

```bash
export LD_LIBRARY_PATH=/path/to/lightgbm/lib:$LD_LIBRARY_PATH
```

### Undefined reference errors

Make sure you're linking against the correct LightGBM library:

```bash
# Check library location
pkg-config --libs lightgbm

# Or manually specify
-L/usr/local/lib -llightgbm
```

## License

MIT License

## Acknowledgments

- [LightGBM](https://github.com/microsoft/LightGBM) by Microsoft
