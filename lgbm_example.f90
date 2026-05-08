!===============================================================================
! LightGBM Fortran Example: Binary Classification
!
! This example demonstrates how to use the LightGBM Fortran API module
! to train a gradient boosting model for binary classification.
!
! Compile with:
!   ifx -o lgbm_example lightgbm_api.f90 lgbm_example.f90 -L/path/to/lightgbm/lib -llightgbm
! or:
!   gfortran -o lgbm_example lightgbm_api.f90 lgbm_example.f90 -L/path/to/lightgbm/lib -llightgbm
!===============================================================================
program lgbm_example
    use lightgbm_api
    use iso_c_binding
    implicit none

    ! Data dimensions
    integer, parameter :: n_train = 1000
    integer, parameter :: n_test = 200
    integer, parameter :: n_features = 10

    ! Data arrays
    real(c_double), allocatable, target :: X_train(:,:), X_test(:,:)
    real(c_double), allocatable, target :: y_train(:), y_test(:)
    real(c_double), allocatable :: predictions(:), eval_results(:)

    ! LightGBM handles
    type(lgbm_dataset_handle) :: train_dataset, valid_dataset
    type(lgbm_booster_handle) :: booster

    ! Parameters
    character(len=512) :: params
    integer :: i, iter, ret
    logical :: is_finished
    real(c_double) :: accuracy

    print '(A)', '========================================'
    print '(A)', ' LightGBM Fortran API Example'
    print '(A)', '========================================'
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Step 1: Generate synthetic data
    !---------------------------------------------------------------------------
    print '(A)', 'Generating synthetic data...'
    call generate_synthetic_data(X_train, y_train, n_train, n_features)
    call generate_synthetic_data(X_test, y_test, n_test, n_features)

    print '(A,I6,A,I3,A)', '  Training samples: ', n_train, ' x ', n_features, ' features'
    print '(A,I6,A,I3,A)', '  Test samples:     ', n_test, ' x ', n_features, ' features'
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Step 2: Create datasets
    !---------------------------------------------------------------------------
    print '(A)', 'Creating LightGBM datasets...'

    ! Create training dataset
    train_dataset = lgbm_dataset_create_from_mat_f64(X_train, "")

    ! Set labels for training data
    call lgbm_dataset_set_label_f64(train_dataset, y_train)

    print '(A,I6)', '  Training dataset: ', lgbm_dataset_get_num_data(train_dataset)
    print '(A,I3)', '  Number of features: ', lgbm_dataset_get_num_feature(train_dataset)

    ! Create validation dataset (with reference to training dataset)
    valid_dataset = lgbm_dataset_create_from_mat_f64(X_test, "", train_dataset)
    call lgbm_dataset_set_label_f64(valid_dataset, y_test)

    print '(A,I6)', '  Validation dataset: ', lgbm_dataset_get_num_data(valid_dataset)
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Step 3: Create and train booster
    !---------------------------------------------------------------------------
    print '(A)', 'Training LightGBM model...'

    ! Set parameters
    params = "objective=binary " // &
             "metric=binary_logloss,auc " // &
             "num_leaves=31 " // &
             "learning_rate=0.05 " // &
             "feature_fraction=0.9 " // &
             "bagging_fraction=0.8 " // &
             "bagging_freq=5 " // &
             "verbose=-1"

    ! Create booster
    booster = lgbm_booster_create(train_dataset, params)

    ! Add validation data
    call lgbm_booster_add_valid_data(booster, valid_dataset)

    ! Training loop
    print '(A)', ''
    print '(A)', '  Iter    Train Loss    Valid Loss    Valid AUC'
    print '(A)', '  ----    ----------    ----------    ---------'

    do iter = 1, 100
        is_finished = lgbm_booster_update_one_iter(booster)
        if (is_finished) exit

        ! Print progress every 10 iterations
        if (mod(iter, 10) == 0) then
            call lgbm_booster_get_eval(booster, 0, eval_results)
            print '(A,I4,A,F12.6)', '  ', iter, '    ', eval_results(1)
            deallocate(eval_results)
        end if
    end do

    ! Get final evaluation
    call lgbm_booster_get_eval(booster, 1, eval_results)
    print '(A)', ''
    print '(A,F10.6)', '  Final validation logloss: ', eval_results(1)
    if (size(eval_results) > 1) then
        print '(A,F10.6)', '  Final validation AUC:     ', eval_results(2)
    end if
    deallocate(eval_results)
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Step 4: Make predictions
    !---------------------------------------------------------------------------
    print '(A)', 'Making predictions on test data...'

    call lgbm_booster_predict_for_mat_f64(booster, X_test, predictions)

    ! Calculate accuracy (threshold at 0.5)
    accuracy = 0.0d0
    do i = 1, n_test
        if ((predictions(i) >= 0.5d0 .and. y_test(i) == 1.0d0) .or. &
            (predictions(i) < 0.5d0 .and. y_test(i) == 0.0d0)) then
            accuracy = accuracy + 1.0d0
        end if
    end do
    accuracy = accuracy / real(n_test, c_double) * 100.0d0

    print '(A,F6.2,A)', '  Test accuracy: ', accuracy, '%'
    print '(A)', ''

    ! Show some predictions
    print '(A)', '  Sample predictions (first 10):'
    print '(A)', '    True    Predicted    Probability'
    do i = 1, min(10, n_test)
        print '(A,F5.1,A,F5.1,A,F10.4)', '    ', y_test(i), '       ', &
            merge(1.0d0, 0.0d0, predictions(i) >= 0.5d0), '         ', predictions(i)
    end do
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Step 5: Save model
    !---------------------------------------------------------------------------
    print '(A)', 'Saving model...'
    call lgbm_booster_save_model(booster, "lgbm_model.txt")
    print '(A)', '  Model saved to: lgbm_model.txt'
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Step 6: Clean up
    !---------------------------------------------------------------------------
    print '(A)', 'Cleaning up...'
    call lgbm_booster_free(booster)
    call lgbm_dataset_free(valid_dataset)
    call lgbm_dataset_free(train_dataset)

    if (allocated(X_train)) deallocate(X_train)
    if (allocated(X_test)) deallocate(X_test)
    if (allocated(y_train)) deallocate(y_train)
    if (allocated(y_test)) deallocate(y_test)
    if (allocated(predictions)) deallocate(predictions)

    print '(A)', '  Done!'
    print '(A)', ''
    print '(A)', '========================================'

contains

    !---------------------------------------------------------------------------
    ! Generate synthetic binary classification data
    !---------------------------------------------------------------------------
    subroutine generate_synthetic_data(X, y, n_samples, n_features)
        real(c_double), allocatable, intent(out) :: X(:,:), y(:)
        integer, intent(in) :: n_samples, n_features

        real(c_double) :: score
        integer :: i, j, seed_size
        integer, allocatable :: seed(:)

        ! Initialize random seed
        call random_seed(size=seed_size)
        allocate(seed(seed_size))
        seed = 42
        call random_seed(put=seed)
        deallocate(seed)

        ! Allocate arrays
        allocate(X(n_samples, n_features))
        allocate(y(n_samples))

        ! Generate features
        do j = 1, n_features
            do i = 1, n_samples
                call random_number(X(i, j))
                X(i, j) = X(i, j) * 2.0d0 - 1.0d0  ! Scale to [-1, 1]
            end do
        end do

        ! Generate labels based on a simple rule
        do i = 1, n_samples
            score = 0.0d0
            do j = 1, min(5, n_features)
                score = score + X(i, j) * (1.0d0 / real(j, c_double))
            end do
            ! Add some non-linearity
            score = score + X(i, 1) * X(i, 2) * 0.5d0

            if (score > 0.0d0) then
                y(i) = 1.0d0
            else
                y(i) = 0.0d0
            end if
        end do
    end subroutine generate_synthetic_data

end program lgbm_example
