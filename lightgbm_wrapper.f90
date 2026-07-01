!===============================================================================
! LightGBM Fortran High-Level Wrapper Module
!
! This module provides a more user-friendly interface to LightGBM,
! with automatic memory management and simpler function signatures.
!===============================================================================
module lightgbm_wrapper
    use lightgbm_api
    use iso_c_binding
    implicit none
    private

    public :: lgbm_model
    public :: lgbm_train
    public :: lgbm_predict
    public :: lgbm_load_model
    public :: lgbm_cross_validate

    !---------------------------------------------------------------------------
    ! High-level model type with automatic resource management
    !---------------------------------------------------------------------------
    type :: lgbm_model
        type(lgbm_booster_handle) :: booster
        type(lgbm_dataset_handle) :: train_data
        type(lgbm_dataset_handle) :: valid_data
        integer :: n_features = 0
        integer :: n_iterations = 0
        logical :: is_trained = .false.
        character(len=32) :: objective = "regression"
    contains
        procedure :: train => model_train
        procedure :: predict => model_predict
        procedure :: save => model_save
        procedure :: load => model_load
        procedure :: free => model_free
        procedure :: get_feature_importance => model_feature_importance
        final :: model_finalizer
    end type lgbm_model

    !---------------------------------------------------------------------------
    ! Training parameters type
    !---------------------------------------------------------------------------
    type, public :: lgbm_params
        character(len=32) :: objective = "regression"
        character(len=64) :: metric = "rmse"
        integer :: num_leaves = 31
        integer :: max_depth = -1
        real(c_double) :: learning_rate = 0.1d0
        real(c_double) :: feature_fraction = 1.0d0
        real(c_double) :: bagging_fraction = 1.0d0
        integer :: bagging_freq = 0
        real(c_double) :: lambda_l1 = 0.0d0
        real(c_double) :: lambda_l2 = 0.0d0
        integer :: min_data_in_leaf = 20
        integer :: num_iterations = 100
        integer :: early_stopping_rounds = 0
        integer :: verbose = 1
        integer :: num_threads = 0
        integer :: seed = 0
        character(len=512) :: monotone_constraints = ""   ! "" = off; else per-feature "0,1,-1,...": LightGBM monotone_constraints
    contains
        procedure :: to_string => params_to_string
    end type lgbm_params

contains

    !---------------------------------------------------------------------------
    ! Convert parameters to LightGBM parameter string
    !---------------------------------------------------------------------------
    function params_to_string(this) result(param_str)
        class(lgbm_params), intent(in) :: this
        character(len=1024) :: param_str
        character(len=32) :: val

        param_str = "objective=" // trim(this%objective) // &
                    " metric=" // trim(this%metric)

        write(val, '(I0)') this%num_leaves
        param_str = trim(param_str) // " num_leaves=" // trim(adjustl(val))

        write(val, '(I0)') this%max_depth
        param_str = trim(param_str) // " max_depth=" // trim(adjustl(val))

        write(val, '(F8.6)') this%learning_rate
        param_str = trim(param_str) // " learning_rate=" // trim(adjustl(val))

        write(val, '(F6.4)') this%feature_fraction
        param_str = trim(param_str) // " feature_fraction=" // trim(adjustl(val))

        write(val, '(F6.4)') this%bagging_fraction
        param_str = trim(param_str) // " bagging_fraction=" // trim(adjustl(val))

        write(val, '(I0)') this%bagging_freq
        param_str = trim(param_str) // " bagging_freq=" // trim(adjustl(val))

        write(val, '(F10.6)') this%lambda_l1
        param_str = trim(param_str) // " lambda_l1=" // trim(adjustl(val))

        write(val, '(F10.6)') this%lambda_l2
        param_str = trim(param_str) // " lambda_l2=" // trim(adjustl(val))

        write(val, '(I0)') this%min_data_in_leaf
        param_str = trim(param_str) // " min_data_in_leaf=" // trim(adjustl(val))

        if (this%num_threads > 0) then
            write(val, '(I0)') this%num_threads
            param_str = trim(param_str) // " num_threads=" // trim(adjustl(val))
        end if

        if (this%seed /= 0) then
            write(val, '(I0)') this%seed
            param_str = trim(param_str) // " seed=" // trim(adjustl(val))
        end if

        if (this%verbose <= 0) then
            param_str = trim(param_str) // " verbose=-1"
        end if

        if (len_trim(this%monotone_constraints) > 0) then
            param_str = trim(param_str) // " monotone_constraints=" // trim(adjustl(this%monotone_constraints))
        end if

    end function params_to_string

    !---------------------------------------------------------------------------
    ! Train model (method)
    !---------------------------------------------------------------------------
    subroutine model_train(this, X, y, X_valid, y_valid, params, &
            train_loss, valid_loss, sample_weight, sample_weight_valid)
        class(lgbm_model), intent(inout) :: this
        real(c_double), intent(in), target :: X(:,:)
        real(c_double), intent(in), target :: y(:)
        real(c_double), intent(in), target, optional :: X_valid(:,:)
        real(c_double), intent(in), target, optional :: y_valid(:)
        type(lgbm_params), intent(in), optional :: params
        real(c_double), intent(out), optional :: train_loss(:)
        real(c_double), intent(out), optional :: valid_loss(:)
        real(c_double), intent(in), target, optional :: sample_weight(:)
        real(c_double), intent(in), target, optional :: sample_weight_valid(:)

        type(lgbm_params) :: p
        character(len=1024) :: param_str
        integer :: iter, ret
        logical :: is_finished, has_valid
        integer(c_int) :: weight_ret
        real(c_float), allocatable, target :: weight_f32(:), weight_valid_f32(:)
        real(c_double), allocatable :: eval_results(:)

        ! Use default or provided parameters
        if (present(params)) then
            p = params
        end if

        param_str = p%to_string()
        this%objective = p%objective
        this%n_features = size(X, 2)

        ! Free existing resources
        call this%free()

        ! Create training dataset
        this%train_data = lgbm_dataset_create_from_mat_f64(X, "")
        call lgbm_dataset_set_label_f64(this%train_data, y)

        ! Optional per-sample weights (e.g., recency decay for time-series).
        ! LightGBM scales each row's loss-gradient contribution by its weight.
        ! NOTE: LightGBM's LGBM_DatasetSetField only supports FLOAT32 for the
        ! "weight" field (FLOAT64 path handles only "init_score" + "label").
        ! Cast our DOUBLE weights down to single precision before passing.
        if (present(sample_weight)) then
            allocate(weight_f32(size(sample_weight)))
            weight_f32 = real(sample_weight, c_float)
            weight_ret = lgbm_dataset_set_field(this%train_data, "weight", &
                c_loc(weight_f32), C_API_DTYPE_FLOAT32)
            ! weight_f32 stays allocated for the rest of this subroutine;
            ! LightGBM copies the buffer during set_field per c_api semantics.
        end if

        ! Check for validation data
        has_valid = present(X_valid) .and. present(y_valid)
        if (has_valid) then
            this%valid_data = lgbm_dataset_create_from_mat_f64(X_valid, "", this%train_data)
            call lgbm_dataset_set_label_f64(this%valid_data, y_valid)
            if (present(sample_weight_valid)) then
                allocate(weight_valid_f32(size(sample_weight_valid)))
                weight_valid_f32 = real(sample_weight_valid, c_float)
                weight_ret = lgbm_dataset_set_field(this%valid_data, "weight", &
                    c_loc(weight_valid_f32), C_API_DTYPE_FLOAT32)
            end if
        end if

        ! Create booster
        this%booster = lgbm_booster_create(this%train_data, param_str)

        if (has_valid) then
            call lgbm_booster_add_valid_data(this%booster, this%valid_data)
        end if

        ! Training loop
        do iter = 1, p%num_iterations
            is_finished = lgbm_booster_update_one_iter(this%booster)
            if (is_finished) exit

            ! Store losses if requested (nested ifs: Fortran .and. does not short-circuit)
            if (present(train_loss)) then
                if (iter <= size(train_loss)) then
                    call lgbm_booster_get_eval(this%booster, 0, eval_results)
                    train_loss(iter) = eval_results(1)
                    deallocate(eval_results)
                end if
            end if

            if (present(valid_loss) .and. has_valid) then
                if (iter <= size(valid_loss)) then
                    call lgbm_booster_get_eval(this%booster, 1, eval_results)
                    valid_loss(iter) = eval_results(1)
                    deallocate(eval_results)
                end if
            end if

            ! Verbose output
            if (p%verbose > 0 .and. mod(iter, 10) == 0) then
                call lgbm_booster_get_eval(this%booster, 0, eval_results)
                if (has_valid) then
                    write(*,'(A,I4,A,F12.6)', advance='no') '[', iter, '] train: ', eval_results(1)
                    deallocate(eval_results)
                    call lgbm_booster_get_eval(this%booster, 1, eval_results)
                    write(*,'(A,F12.6)') '  valid: ', eval_results(1)
                else
                    write(*,'(A,I4,A,F12.6)') '[', iter, '] train: ', eval_results(1)
                end if
                deallocate(eval_results)
            end if
        end do

        this%n_iterations = lgbm_booster_get_current_iteration(this%booster)
        this%is_trained = .true.
    end subroutine model_train

    !---------------------------------------------------------------------------
    ! Predict (method)
    !---------------------------------------------------------------------------
    subroutine model_predict(this, X, predictions, predict_type)
        class(lgbm_model), intent(in) :: this
        real(c_double), intent(in), target :: X(:,:)
        real(c_double), intent(out), allocatable :: predictions(:)
        integer, intent(in), optional :: predict_type

        if (.not. this%is_trained) then
            print '(A)', 'Error: Model has not been trained yet!'
            return
        end if

        call lgbm_booster_predict_for_mat_f64(this%booster, X, predictions, predict_type)
    end subroutine model_predict

    !---------------------------------------------------------------------------
    ! Save model (method)
    !---------------------------------------------------------------------------
    subroutine model_save(this, filename)
        class(lgbm_model), intent(in) :: this
        character(len=*), intent(in) :: filename

        if (.not. this%is_trained) then
            print '(A)', 'Error: Model has not been trained yet!'
            return
        end if

        call lgbm_booster_save_model(this%booster, filename)
    end subroutine model_save

    !---------------------------------------------------------------------------
    ! Load model (method)
    !---------------------------------------------------------------------------
    subroutine model_load(this, filename)
        class(lgbm_model), intent(inout) :: this
        character(len=*), intent(in) :: filename

        call this%free()
        this%booster = lgbm_booster_create_from_model_file(filename, this%n_iterations)
        this%is_trained = .true.
    end subroutine model_load

    !---------------------------------------------------------------------------
    ! Free resources (method)
    !---------------------------------------------------------------------------
    subroutine model_free(this)
        class(lgbm_model), intent(inout) :: this

        if (this%booster%is_valid()) call lgbm_booster_free(this%booster)
        if (this%valid_data%is_valid()) call lgbm_dataset_free(this%valid_data)
        if (this%train_data%is_valid()) call lgbm_dataset_free(this%train_data)

        this%is_trained = .false.
        this%n_iterations = 0
    end subroutine model_free

    !---------------------------------------------------------------------------
    ! Finalizer (automatically called on deallocation)
    !---------------------------------------------------------------------------
    subroutine model_finalizer(this)
        type(lgbm_model), intent(inout) :: this
        call this%free()
    end subroutine model_finalizer

    !---------------------------------------------------------------------------
    ! Get feature importance (method)
    !---------------------------------------------------------------------------
    subroutine model_feature_importance(this, importance, importance_type)
        class(lgbm_model), intent(in) :: this
        real(c_double), intent(out), allocatable :: importance(:)
        integer, intent(in), optional :: importance_type

        integer :: imp_type

        imp_type = C_API_FEATURE_IMPORTANCE_SPLIT
        if (present(importance_type)) imp_type = importance_type

        if (.not. this%is_trained) then
            print '(A)', 'Error: Model has not been trained yet!'
            return
        end if

        call lgbm_booster_feature_importance(this%booster, importance, &
            this%n_iterations, imp_type)

        ! Resize to actual number of features
        if (size(importance) > this%n_features) then
            importance = importance(1:this%n_features)
        end if
    end subroutine model_feature_importance

    !---------------------------------------------------------------------------
    ! Convenience function: Train model
    !---------------------------------------------------------------------------
    function lgbm_train(X, y, X_valid, y_valid, params) result(model)
        real(c_double), intent(in), target :: X(:,:)
        real(c_double), intent(in), target :: y(:)
        real(c_double), intent(in), target, optional :: X_valid(:,:)
        real(c_double), intent(in), target, optional :: y_valid(:)
        type(lgbm_params), intent(in), optional :: params
        type(lgbm_model) :: model

        if (present(X_valid) .and. present(y_valid)) then
            if (present(params)) then
                call model%train(X, y, X_valid, y_valid, params)
            else
                call model%train(X, y, X_valid, y_valid)
            end if
        else
            if (present(params)) then
                call model%train(X, y, params=params)
            else
                call model%train(X, y)
            end if
        end if
    end function lgbm_train

    !---------------------------------------------------------------------------
    ! Convenience function: Predict
    !---------------------------------------------------------------------------
    subroutine lgbm_predict(model, X, predictions)
        type(lgbm_model), intent(in) :: model
        real(c_double), intent(in), target :: X(:,:)
        real(c_double), intent(out), allocatable :: predictions(:)

        call model%predict(X, predictions)
    end subroutine lgbm_predict

    !---------------------------------------------------------------------------
    ! Convenience function: Load model
    !---------------------------------------------------------------------------
    function lgbm_load_model(filename) result(model)
        character(len=*), intent(in) :: filename
        type(lgbm_model) :: model

        call model%load(filename)
    end function lgbm_load_model

    !---------------------------------------------------------------------------
    ! K-Fold Cross-Validation
    !---------------------------------------------------------------------------
    subroutine lgbm_cross_validate(X, y, params, n_folds, scores, score_std)
        real(c_double), intent(in), target :: X(:,:)
        real(c_double), intent(in), target :: y(:)
        type(lgbm_params), intent(in) :: params
        integer, intent(in) :: n_folds
        real(c_double), intent(out) :: scores(n_folds)
        real(c_double), intent(out), optional :: score_std

        integer :: n_samples, fold_size, fold
        integer :: train_start, train_end, valid_start, valid_end
        integer :: i, j, n_train
        real(c_double), allocatable :: X_train(:,:), y_train(:)
        real(c_double), allocatable :: X_valid(:,:), y_valid(:)
        real(c_double), allocatable :: predictions(:), eval_results(:)
        type(lgbm_model) :: model
        type(lgbm_params) :: cv_params
        real(c_double) :: mean_score

        n_samples = size(X, 1)
        fold_size = n_samples / n_folds

        cv_params = params
        cv_params%verbose = 0

        do fold = 1, n_folds
            ! Define validation indices for this fold
            valid_start = (fold - 1) * fold_size + 1
            if (fold == n_folds) then
                valid_end = n_samples
            else
                valid_end = fold * fold_size
            end if

            ! Allocate validation data
            allocate(X_valid(valid_end - valid_start + 1, size(X, 2)))
            allocate(y_valid(valid_end - valid_start + 1))

            X_valid = X(valid_start:valid_end, :)
            y_valid = y(valid_start:valid_end)

            ! Allocate training data (all except validation fold)
            n_train = n_samples - (valid_end - valid_start + 1)
            allocate(X_train(n_train, size(X, 2)))
            allocate(y_train(n_train))

            j = 1
            do i = 1, n_samples
                if (i < valid_start .or. i > valid_end) then
                    X_train(j, :) = X(i, :)
                    y_train(j) = y(i)
                    j = j + 1
                end if
            end do

            ! Train model on this fold
            call model%train(X_train, y_train, X_valid, y_valid, cv_params)

            ! Get validation score
            call lgbm_booster_get_eval(model%booster, 1, eval_results)
            scores(fold) = eval_results(1)
            deallocate(eval_results)

            ! Clean up
            call model%free()
            deallocate(X_train, y_train, X_valid, y_valid)

            print '(A,I2,A,F12.6)', '  Fold ', fold, ' score: ', scores(fold)
        end do

        ! Calculate mean and std
        mean_score = sum(scores) / real(n_folds, c_double)

        if (present(score_std)) then
            score_std = sqrt(sum((scores - mean_score)**2) / real(n_folds - 1, c_double))
        end if

        print '(A)'
        print '(A,F12.6,A,F12.6)', '  CV Mean: ', mean_score, ' +/- ', &
            sqrt(sum((scores - mean_score)**2) / real(n_folds - 1, c_double))
    end subroutine lgbm_cross_validate

end module lightgbm_wrapper
