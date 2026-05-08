!===============================================================================
! LightGBM Fortran High-Level Wrapper Example
!
! This example demonstrates the simplified high-level API for LightGBM.
!
! Compile with:
!   ifx -o lgbm_wrapper_example lightgbm_api.f90 lightgbm_wrapper.f90 \
!       lgbm_wrapper_example.f90 -L/path/to/lightgbm/lib -llightgbm
!===============================================================================
program lgbm_wrapper_example
    use lightgbm_wrapper
    use lightgbm_api, only: C_API_FEATURE_IMPORTANCE_GAIN
    use iso_c_binding
    implicit none

    ! Data
    integer, parameter :: n_samples = 1000
    integer, parameter :: n_features = 10
    integer, parameter :: n_train = 800
    integer, parameter :: n_test = 200

    real(c_double), allocatable :: X(:,:), y(:)
    real(c_double), allocatable :: X_train(:,:), y_train(:)
    real(c_double), allocatable :: X_test(:,:), y_test(:)
    real(c_double), allocatable :: predictions(:), importance(:)
    real(c_double) :: cv_scores(5)

    ! Model and parameters
    type(lgbm_model) :: model
    type(lgbm_params) :: params

    integer :: i
    real(c_double) :: rmse, r_squared, y_mean, ss_tot, ss_res

    print '(A)', '========================================'
    print '(A)', ' LightGBM High-Level Wrapper Example'
    print '(A)', '========================================'
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Generate synthetic regression data
    !---------------------------------------------------------------------------
    print '(A)', '1. Generating synthetic data...'

    allocate(X(n_samples, n_features))
    allocate(y(n_samples))

    call random_number(X)

    ! Create non-linear target with noise
    y = 3.0d0 * X(:, 1) + &
        2.0d0 * X(:, 2)**2 + &
        1.5d0 * sin(X(:, 3) * 6.28d0) + &
        X(:, 4) * X(:, 5) + &
        0.1d0 * X(:, 10)  ! noise feature

    ! Add some random noise
    do i = 1, n_samples
        call random_number(rmse)
        y(i) = y(i) + (rmse - 0.5d0) * 0.2d0
    end do

    ! Split data
    allocate(X_train(n_train, n_features))
    allocate(y_train(n_train))
    allocate(X_test(n_test, n_features))
    allocate(y_test(n_test))

    X_train = X(1:n_train, :)
    y_train = y(1:n_train)
    X_test = X(n_train+1:n_samples, :)
    y_test = y(n_train+1:n_samples)

    print '(A,I4,A)', '   Generated ', n_samples, ' samples'
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Configure parameters
    !---------------------------------------------------------------------------
    print '(A)', '2. Configuring model parameters...'

    params%objective = "regression"
    params%metric = "rmse"
    params%num_leaves = 31
    params%max_depth = 6
    params%learning_rate = 0.05d0
    params%feature_fraction = 0.8d0
    params%bagging_fraction = 0.8d0
    params%bagging_freq = 5
    params%lambda_l2 = 0.1d0
    params%min_data_in_leaf = 10
    params%num_iterations = 200
    params%verbose = 1
    params%seed = 42

    print '(A)', '   Parameters set:'
    print '(A,A)', '     objective     = ', trim(params%objective)
    print '(A,I3)', '     num_leaves    = ', params%num_leaves
    print '(A,F6.4)', '     learning_rate = ', params%learning_rate
    print '(A,I4)', '     iterations    = ', params%num_iterations
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Train model
    !---------------------------------------------------------------------------
    print '(A)', '3. Training model...'
    print '(A)', ''

    call model%train(X_train, y_train, X_test, y_test, params)

    print '(A)', ''
    print '(A,I4,A)', '   Trained for ', model%n_iterations, ' iterations'
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Make predictions
    !---------------------------------------------------------------------------
    print '(A)', '4. Making predictions...'

    call model%predict(X_test, predictions)

    ! Calculate RMSE
    rmse = sqrt(sum((predictions - y_test)**2) / real(n_test, c_double))

    ! Calculate R²
    y_mean = sum(y_test) / real(n_test, c_double)
    ss_tot = sum((y_test - y_mean)**2)
    ss_res = sum((y_test - predictions)**2)
    r_squared = 1.0d0 - ss_res / ss_tot

    print '(A,F10.6)', '   Test RMSE: ', rmse
    print '(A,F10.6)', '   Test R²:   ', r_squared
    print '(A)', ''

    ! Show sample predictions
    print '(A)', '   Sample predictions (first 5):'
    print '(A)', '     Actual      Predicted'
    do i = 1, 5
        print '(A,F10.4,A,F10.4)', '     ', y_test(i), '    ', predictions(i)
    end do
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Feature importance
    !---------------------------------------------------------------------------
    print '(A)', '5. Feature importance...'

    call model%get_feature_importance(importance, C_API_FEATURE_IMPORTANCE_GAIN)

    print '(A)', '   Feature    Importance (Gain)'
    do i = 1, n_features
        print '(A,I2,A,F12.2)', '      ', i, '        ', importance(i)
    end do
    print '(A)', ''
    deallocate(importance)

    !---------------------------------------------------------------------------
    ! Save and reload model
    !---------------------------------------------------------------------------
    print '(A)', '6. Save and reload model...'

    call model%save("lgbm_regression_model.txt")
    print '(A)', '   Model saved to: lgbm_regression_model.txt'

    ! Free original model
    call model%free()

    ! Reload model
    call model%load("lgbm_regression_model.txt")
    print '(A)', '   Model reloaded successfully'

    ! Verify predictions still work
    deallocate(predictions)
    call model%predict(X_test, predictions)
    rmse = sqrt(sum((predictions - y_test)**2) / real(n_test, c_double))
    print '(A,F10.6)', '   Reloaded model RMSE: ', rmse
    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Cross-validation
    !---------------------------------------------------------------------------
    print '(A)', '7. 5-Fold Cross-Validation...'
    print '(A)', ''

    call lgbm_cross_validate(X, y, params, 5, cv_scores)

    print '(A)', ''

    !---------------------------------------------------------------------------
    ! Clean up
    !---------------------------------------------------------------------------
    print '(A)', '8. Cleaning up...'

    call model%free()  ! Or let finalizer handle it

    deallocate(X, y, X_train, y_train, X_test, y_test, predictions)

    print '(A)', '   Done!'
    print '(A)', ''
    print '(A)', '========================================'

end program lgbm_wrapper_example
