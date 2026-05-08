!===============================================================================
! LightGBM Fortran API Module
! 
! This module provides Fortran bindings to the LightGBM C API using ISO_C_BINDING.
! It allows training gradient boosting models directly from Fortran code.
!
! Requirements:
!   - LightGBM library (liblightgbm.so / lightgbm.dll)
!   - Fortran 2003+ compiler with ISO_C_BINDING support
!
! Author: Claude
! Date: 2025
!===============================================================================
module lightgbm_api
    use, intrinsic :: iso_c_binding
    implicit none
    private

    !---------------------------------------------------------------------------
    ! Public interfaces
    !---------------------------------------------------------------------------
    public :: lgbm_dataset_handle, lgbm_booster_handle
    public :: lgbm_dataset_create_from_mat
    public :: lgbm_dataset_create_from_mat_f64
    public :: lgbm_dataset_create_from_mat_f32
    public :: lgbm_dataset_set_field
    public :: lgbm_dataset_set_label_f32
    public :: lgbm_dataset_set_label_f64
    public :: lgbm_dataset_get_num_data
    public :: lgbm_dataset_get_num_feature
    public :: lgbm_dataset_free
    public :: lgbm_booster_create
    public :: lgbm_booster_create_from_model_file
    public :: lgbm_booster_free
    public :: lgbm_booster_add_valid_data
    public :: lgbm_booster_update_one_iter
    public :: lgbm_booster_get_eval
    public :: lgbm_booster_get_current_iteration
    public :: lgbm_booster_predict_for_mat
    public :: lgbm_booster_predict_for_mat_f64
    public :: lgbm_booster_predict_for_mat_f32
    public :: lgbm_booster_save_model
    public :: lgbm_booster_dump_model
    public :: lgbm_booster_feature_importance
    public :: lgbm_get_last_error
    public :: lgbm_check_error

    ! Data type constants
    public :: C_API_DTYPE_FLOAT32, C_API_DTYPE_FLOAT64
    public :: C_API_DTYPE_INT32, C_API_DTYPE_INT64
    public :: C_API_PREDICT_NORMAL, C_API_PREDICT_RAW_SCORE
    public :: C_API_PREDICT_LEAF_INDEX, C_API_PREDICT_CONTRIB
    public :: C_API_FEATURE_IMPORTANCE_SPLIT, C_API_FEATURE_IMPORTANCE_GAIN

    !---------------------------------------------------------------------------
    ! Constants
    !---------------------------------------------------------------------------
    ! Data types
    integer(c_int), parameter :: C_API_DTYPE_FLOAT32 = 0
    integer(c_int), parameter :: C_API_DTYPE_FLOAT64 = 1
    integer(c_int), parameter :: C_API_DTYPE_INT32   = 2
    integer(c_int), parameter :: C_API_DTYPE_INT64   = 3

    ! Prediction types
    integer(c_int), parameter :: C_API_PREDICT_NORMAL      = 0
    integer(c_int), parameter :: C_API_PREDICT_RAW_SCORE   = 1
    integer(c_int), parameter :: C_API_PREDICT_LEAF_INDEX  = 2
    integer(c_int), parameter :: C_API_PREDICT_CONTRIB     = 3

    ! Feature importance types
    integer(c_int), parameter :: C_API_FEATURE_IMPORTANCE_SPLIT = 0
    integer(c_int), parameter :: C_API_FEATURE_IMPORTANCE_GAIN  = 1

    !---------------------------------------------------------------------------
    ! Handle types (opaque pointers)
    !---------------------------------------------------------------------------
    type :: lgbm_dataset_handle
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => dataset_is_valid
    end type lgbm_dataset_handle

    type :: lgbm_booster_handle
        type(c_ptr) :: ptr = c_null_ptr
    contains
        procedure :: is_valid => booster_is_valid
    end type lgbm_booster_handle

    !---------------------------------------------------------------------------
    ! C API Interface declarations
    !---------------------------------------------------------------------------
    interface

        ! Get last error message
        function LGBM_GetLastError() bind(C, name='LGBM_GetLastError')
            import :: c_ptr
            type(c_ptr) :: LGBM_GetLastError
        end function LGBM_GetLastError

        ! Create dataset from dense matrix
        function LGBM_DatasetCreateFromMat(data, data_type, nrow, ncol, is_row_major, &
                parameters, reference, out) bind(C, name='LGBM_DatasetCreateFromMat')
            import :: c_ptr, c_int, c_int32_t, c_char
            type(c_ptr), value :: data
            integer(c_int), value :: data_type
            integer(c_int32_t), value :: nrow
            integer(c_int32_t), value :: ncol
            integer(c_int), value :: is_row_major
            character(kind=c_char), intent(in) :: parameters(*)
            type(c_ptr), value :: reference
            type(c_ptr), intent(out) :: out
            integer(c_int) :: LGBM_DatasetCreateFromMat
        end function LGBM_DatasetCreateFromMat

        ! Set field in dataset (e.g., label, weight)
        function LGBM_DatasetSetField(handle, field_name, field_data, num_element, &
                dtype) bind(C, name='LGBM_DatasetSetField')
            import :: c_ptr, c_int, c_int32_t, c_char
            type(c_ptr), value :: handle
            character(kind=c_char), intent(in) :: field_name(*)
            type(c_ptr), value :: field_data
            integer(c_int32_t), value :: num_element
            integer(c_int), value :: dtype
            integer(c_int) :: LGBM_DatasetSetField
        end function LGBM_DatasetSetField

        ! Get number of data points
        function LGBM_DatasetGetNumData(handle, out) bind(C, name='LGBM_DatasetGetNumData')
            import :: c_ptr, c_int, c_int32_t
            type(c_ptr), value :: handle
            integer(c_int32_t), intent(out) :: out
            integer(c_int) :: LGBM_DatasetGetNumData
        end function LGBM_DatasetGetNumData

        ! Get number of features
        function LGBM_DatasetGetNumFeature(handle, out) bind(C, name='LGBM_DatasetGetNumFeature')
            import :: c_ptr, c_int, c_int32_t
            type(c_ptr), value :: handle
            integer(c_int32_t), intent(out) :: out
            integer(c_int) :: LGBM_DatasetGetNumFeature
        end function LGBM_DatasetGetNumFeature

        ! Free dataset
        function LGBM_DatasetFree(handle) bind(C, name='LGBM_DatasetFree')
            import :: c_ptr, c_int
            type(c_ptr), value :: handle
            integer(c_int) :: LGBM_DatasetFree
        end function LGBM_DatasetFree

        ! Create booster
        function LGBM_BoosterCreate(train_data, parameters, out) &
                bind(C, name='LGBM_BoosterCreate')
            import :: c_ptr, c_int, c_char
            type(c_ptr), value :: train_data
            character(kind=c_char), intent(in) :: parameters(*)
            type(c_ptr), intent(out) :: out
            integer(c_int) :: LGBM_BoosterCreate
        end function LGBM_BoosterCreate

        ! Create booster from model file
        function LGBM_BoosterCreateFromModelfile(filename, out_num_iterations, out) &
                bind(C, name='LGBM_BoosterCreateFromModelfile')
            import :: c_ptr, c_int, c_char
            character(kind=c_char), intent(in) :: filename(*)
            integer(c_int), intent(out) :: out_num_iterations
            type(c_ptr), intent(out) :: out
            integer(c_int) :: LGBM_BoosterCreateFromModelfile
        end function LGBM_BoosterCreateFromModelfile

        ! Free booster
        function LGBM_BoosterFree(handle) bind(C, name='LGBM_BoosterFree')
            import :: c_ptr, c_int
            type(c_ptr), value :: handle
            integer(c_int) :: LGBM_BoosterFree
        end function LGBM_BoosterFree

        ! Add validation data
        function LGBM_BoosterAddValidData(handle, valid_data) &
                bind(C, name='LGBM_BoosterAddValidData')
            import :: c_ptr, c_int
            type(c_ptr), value :: handle
            type(c_ptr), value :: valid_data
            integer(c_int) :: LGBM_BoosterAddValidData
        end function LGBM_BoosterAddValidData

        ! Update one iteration
        function LGBM_BoosterUpdateOneIter(handle, is_finished) &
                bind(C, name='LGBM_BoosterUpdateOneIter')
            import :: c_ptr, c_int
            type(c_ptr), value :: handle
            integer(c_int), intent(out) :: is_finished
            integer(c_int) :: LGBM_BoosterUpdateOneIter
        end function LGBM_BoosterUpdateOneIter

        ! Get evaluation results
        function LGBM_BoosterGetEval(handle, data_idx, out_len, out_results) &
                bind(C, name='LGBM_BoosterGetEval')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: handle
            integer(c_int), value :: data_idx
            integer(c_int), intent(out) :: out_len
            type(c_ptr), value :: out_results
            integer(c_int) :: LGBM_BoosterGetEval
        end function LGBM_BoosterGetEval

        ! Get current iteration
        function LGBM_BoosterGetCurrentIteration(handle, out_iteration) &
                bind(C, name='LGBM_BoosterGetCurrentIteration')
            import :: c_ptr, c_int
            type(c_ptr), value :: handle
            integer(c_int), intent(out) :: out_iteration
            integer(c_int) :: LGBM_BoosterGetCurrentIteration
        end function LGBM_BoosterGetCurrentIteration

        ! Predict for dense matrix
        function LGBM_BoosterPredictForMat(handle, data, data_type, nrow, ncol, &
                is_row_major, predict_type, start_iteration, num_iteration, &
                parameters, out_len, out_result) &
                bind(C, name='LGBM_BoosterPredictForMat')
            import :: c_ptr, c_int, c_int32_t, c_int64_t, c_char
            type(c_ptr), value :: handle
            type(c_ptr), value :: data
            integer(c_int), value :: data_type
            integer(c_int32_t), value :: nrow
            integer(c_int32_t), value :: ncol
            integer(c_int), value :: is_row_major
            integer(c_int), value :: predict_type
            integer(c_int), value :: start_iteration
            integer(c_int), value :: num_iteration
            character(kind=c_char), intent(in) :: parameters(*)
            integer(c_int64_t), intent(out) :: out_len
            type(c_ptr), value :: out_result
            integer(c_int) :: LGBM_BoosterPredictForMat
        end function LGBM_BoosterPredictForMat

        ! Save model to file
        function LGBM_BoosterSaveModel(handle, start_iteration, num_iteration, &
                feature_importance_type, filename) &
                bind(C, name='LGBM_BoosterSaveModel')
            import :: c_ptr, c_int, c_char
            type(c_ptr), value :: handle
            integer(c_int), value :: start_iteration
            integer(c_int), value :: num_iteration
            integer(c_int), value :: feature_importance_type
            character(kind=c_char), intent(in) :: filename(*)
            integer(c_int) :: LGBM_BoosterSaveModel
        end function LGBM_BoosterSaveModel

        ! Dump model to string
        function LGBM_BoosterDumpModel(handle, start_iteration, num_iteration, &
                feature_importance_type, buffer_len, out_len, out_str) &
                bind(C, name='LGBM_BoosterDumpModel')
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: handle
            integer(c_int), value :: start_iteration
            integer(c_int), value :: num_iteration
            integer(c_int), value :: feature_importance_type
            integer(c_int64_t), value :: buffer_len
            integer(c_int64_t), intent(out) :: out_len
            character(kind=c_char), intent(out) :: out_str(*)
            integer(c_int) :: LGBM_BoosterDumpModel
        end function LGBM_BoosterDumpModel

        ! Get feature importance
        function LGBM_BoosterFeatureImportance(handle, num_iteration, &
                importance_type, out_results) &
                bind(C, name='LGBM_BoosterFeatureImportance')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: handle
            integer(c_int), value :: num_iteration
            integer(c_int), value :: importance_type
            type(c_ptr), value :: out_results
            integer(c_int) :: LGBM_BoosterFeatureImportance
        end function LGBM_BoosterFeatureImportance

    end interface

contains

    !---------------------------------------------------------------------------
    ! Helper: Check if dataset handle is valid
    !---------------------------------------------------------------------------
    logical function dataset_is_valid(this)
        class(lgbm_dataset_handle), intent(in) :: this
        dataset_is_valid = c_associated(this%ptr)
    end function dataset_is_valid

    !---------------------------------------------------------------------------
    ! Helper: Check if booster handle is valid
    !---------------------------------------------------------------------------
    logical function booster_is_valid(this)
        class(lgbm_booster_handle), intent(in) :: this
        booster_is_valid = c_associated(this%ptr)
    end function booster_is_valid

    !---------------------------------------------------------------------------
    ! Get last error message as Fortran string
    !---------------------------------------------------------------------------
    function lgbm_get_last_error() result(error_msg)
        character(len=:), allocatable :: error_msg
        type(c_ptr) :: c_str
        character(kind=c_char), pointer :: f_ptr(:)
        integer :: i, str_len

        c_str = LGBM_GetLastError()
        if (.not. c_associated(c_str)) then
            error_msg = ""
            return
        end if

        ! Find string length
        str_len = 0
        call c_f_pointer(c_str, f_ptr, [1024])
        do i = 1, 1024
            if (f_ptr(i) == c_null_char) exit
            str_len = str_len + 1
        end do

        ! Copy to Fortran string
        allocate(character(len=str_len) :: error_msg)
        do i = 1, str_len
            error_msg(i:i) = f_ptr(i)
        end do
    end function lgbm_get_last_error

    !---------------------------------------------------------------------------
    ! Check error and print message
    !---------------------------------------------------------------------------
    subroutine lgbm_check_error(ret_code, operation, iostat)
        integer(c_int), intent(in) :: ret_code
        character(len=*), intent(in), optional :: operation
        integer, intent(out), optional :: iostat
        character(len=:), allocatable :: error_msg

        if (present(iostat)) iostat = int(ret_code)

        if (ret_code /= 0) then
            error_msg = lgbm_get_last_error()
            if (present(operation)) then
                write(*,'(A,A,A,A)') "LightGBM Error in ", operation, ": ", error_msg
            else
                write(*,'(A,A)') "LightGBM Error: ", error_msg
            end if
        end if
    end subroutine lgbm_check_error

    !---------------------------------------------------------------------------
    ! Convert Fortran string to C string
    !---------------------------------------------------------------------------
    function to_c_string(f_string) result(c_string)
        character(len=*), intent(in) :: f_string
        character(len=:), allocatable :: c_string
        c_string = trim(f_string) // c_null_char
    end function to_c_string

    !---------------------------------------------------------------------------
    ! Create dataset from matrix (generic interface)
    !---------------------------------------------------------------------------
    function lgbm_dataset_create_from_mat(data_ptr, data_type, nrow, ncol, &
            parameters, reference) result(dataset)
        type(c_ptr), intent(in) :: data_ptr
        integer(c_int), intent(in) :: data_type
        integer, intent(in) :: nrow, ncol
        character(len=*), intent(in) :: parameters
        type(lgbm_dataset_handle), intent(in), optional :: reference
        type(lgbm_dataset_handle) :: dataset

        integer(c_int) :: ret
        type(c_ptr) :: ref_ptr
        character(len=:), allocatable :: c_params

        c_params = to_c_string(parameters)

        if (present(reference)) then
            ref_ptr = reference%ptr
        else
            ref_ptr = c_null_ptr
        end if

        ret = LGBM_DatasetCreateFromMat( &
            data_ptr, &
            data_type, &
            int(nrow, c_int32_t), &
            int(ncol, c_int32_t), &
            0_c_int, &  ! is_row_major=0: Fortran column-major layout
            c_params, &
            ref_ptr, &
            dataset%ptr)

        call lgbm_check_error(ret, "DatasetCreateFromMat")
    end function lgbm_dataset_create_from_mat

    !---------------------------------------------------------------------------
    ! Create dataset from double precision matrix
    !---------------------------------------------------------------------------
    function lgbm_dataset_create_from_mat_f64(data, parameters, reference) result(dataset)
        real(c_double), intent(in), target :: data(:,:)
        character(len=*), intent(in) :: parameters
        type(lgbm_dataset_handle), intent(in), optional :: reference
        type(lgbm_dataset_handle) :: dataset

        integer :: nrow, ncol

        nrow = size(data, 1)
        ncol = size(data, 2)

        if (present(reference)) then
            dataset = lgbm_dataset_create_from_mat(c_loc(data), C_API_DTYPE_FLOAT64, &
                nrow, ncol, parameters, reference)
        else
            dataset = lgbm_dataset_create_from_mat(c_loc(data), C_API_DTYPE_FLOAT64, &
                nrow, ncol, parameters)
        end if
    end function lgbm_dataset_create_from_mat_f64

    !---------------------------------------------------------------------------
    ! Create dataset from single precision matrix
    !---------------------------------------------------------------------------
    function lgbm_dataset_create_from_mat_f32(data, parameters, reference) result(dataset)
        real(c_float), intent(in), target :: data(:,:)
        character(len=*), intent(in) :: parameters
        type(lgbm_dataset_handle), intent(in), optional :: reference
        type(lgbm_dataset_handle) :: dataset

        integer :: nrow, ncol

        nrow = size(data, 1)
        ncol = size(data, 2)

        if (present(reference)) then
            dataset = lgbm_dataset_create_from_mat(c_loc(data), C_API_DTYPE_FLOAT32, &
                nrow, ncol, parameters, reference)
        else
            dataset = lgbm_dataset_create_from_mat(c_loc(data), C_API_DTYPE_FLOAT32, &
                nrow, ncol, parameters)
        end if
    end function lgbm_dataset_create_from_mat_f32

    !---------------------------------------------------------------------------
    ! Set field in dataset (label, weight, etc.)
    !---------------------------------------------------------------------------
    function lgbm_dataset_set_field(dataset, field_name, field_data, data_type) result(ret)
        type(lgbm_dataset_handle), intent(in) :: dataset
        character(len=*), intent(in) :: field_name
        type(c_ptr), intent(in) :: field_data
        integer(c_int), intent(in) :: data_type
        integer :: num_data
        integer(c_int) :: ret

        character(len=:), allocatable :: c_field_name

        c_field_name = to_c_string(field_name)

        ! Get number of data points
        num_data = lgbm_dataset_get_num_data(dataset)

        ret = LGBM_DatasetSetField(dataset%ptr, c_field_name, field_data, &
            int(num_data, c_int32_t), data_type)

        if (ret /= 0) then
            call lgbm_check_error(ret, "DatasetSetField")
        end if
    end function lgbm_dataset_set_field

    !---------------------------------------------------------------------------
    ! Set label field (convenience function for float32)
    !---------------------------------------------------------------------------
    subroutine lgbm_dataset_set_label_f32(dataset, labels)
        type(lgbm_dataset_handle), intent(in) :: dataset
        real(c_float), intent(in), target :: labels(:)
        integer(c_int) :: ret
        character(len=:), allocatable :: c_field_name

        c_field_name = to_c_string("label")
        ret = LGBM_DatasetSetField(dataset%ptr, c_field_name, c_loc(labels), &
            int(size(labels), c_int32_t), C_API_DTYPE_FLOAT32)
        call lgbm_check_error(ret, "DatasetSetLabel")
    end subroutine lgbm_dataset_set_label_f32

    !---------------------------------------------------------------------------
    ! Set label field (convenience function for float64)
    !---------------------------------------------------------------------------
    subroutine lgbm_dataset_set_label_f64(dataset, labels)
        type(lgbm_dataset_handle), intent(in) :: dataset
        real(c_double), intent(in), target :: labels(:)
        integer(c_int) :: ret
        character(len=:), allocatable :: c_field_name
        real(c_float), allocatable, target :: labels_f32(:)

        ! LightGBM requires float32 for labels
        allocate(labels_f32(size(labels)))
        labels_f32 = real(labels, c_float)

        c_field_name = to_c_string("label")
        ret = LGBM_DatasetSetField(dataset%ptr, c_field_name, c_loc(labels_f32), &
            int(size(labels), c_int32_t), C_API_DTYPE_FLOAT32)
        call lgbm_check_error(ret, "DatasetSetLabel")

        deallocate(labels_f32)
    end subroutine lgbm_dataset_set_label_f64

    !---------------------------------------------------------------------------
    ! Get number of data points in dataset
    !---------------------------------------------------------------------------
    function lgbm_dataset_get_num_data(dataset) result(num_data)
        type(lgbm_dataset_handle), intent(in) :: dataset
        integer :: num_data
        integer(c_int32_t) :: out_num
        integer(c_int) :: ret

        ret = LGBM_DatasetGetNumData(dataset%ptr, out_num)
        call lgbm_check_error(ret, "DatasetGetNumData")
        num_data = int(out_num)
    end function lgbm_dataset_get_num_data

    !---------------------------------------------------------------------------
    ! Get number of features in dataset
    !---------------------------------------------------------------------------
    function lgbm_dataset_get_num_feature(dataset) result(num_feature)
        type(lgbm_dataset_handle), intent(in) :: dataset
        integer :: num_feature
        integer(c_int32_t) :: out_num
        integer(c_int) :: ret

        ret = LGBM_DatasetGetNumFeature(dataset%ptr, out_num)
        call lgbm_check_error(ret, "DatasetGetNumFeature")
        num_feature = int(out_num)
    end function lgbm_dataset_get_num_feature

    !---------------------------------------------------------------------------
    ! Free dataset
    !---------------------------------------------------------------------------
    subroutine lgbm_dataset_free(dataset)
        type(lgbm_dataset_handle), intent(inout) :: dataset
        integer(c_int) :: ret

        if (dataset%is_valid()) then
            ret = LGBM_DatasetFree(dataset%ptr)
            call lgbm_check_error(ret, "DatasetFree")
            dataset%ptr = c_null_ptr
        end if
    end subroutine lgbm_dataset_free

    !---------------------------------------------------------------------------
    ! Create booster
    !---------------------------------------------------------------------------
    function lgbm_booster_create(train_data, parameters) result(booster)
        type(lgbm_dataset_handle), intent(in) :: train_data
        character(len=*), intent(in) :: parameters
        type(lgbm_booster_handle) :: booster

        integer(c_int) :: ret
        character(len=:), allocatable :: c_params

        c_params = to_c_string(parameters)

        ret = LGBM_BoosterCreate(train_data%ptr, c_params, booster%ptr)
        call lgbm_check_error(ret, "BoosterCreate")
    end function lgbm_booster_create

    !---------------------------------------------------------------------------
    ! Create booster from model file
    !---------------------------------------------------------------------------
    function lgbm_booster_create_from_model_file(filename, num_iterations) result(booster)
        character(len=*), intent(in) :: filename
        integer, intent(out), optional :: num_iterations
        type(lgbm_booster_handle) :: booster

        integer(c_int) :: ret, out_num_iter
        character(len=:), allocatable :: c_filename

        c_filename = to_c_string(filename)

        ret = LGBM_BoosterCreateFromModelfile(c_filename, out_num_iter, booster%ptr)
        call lgbm_check_error(ret, "BoosterCreateFromModelfile")

        if (present(num_iterations)) then
            num_iterations = int(out_num_iter)
        end if
    end function lgbm_booster_create_from_model_file

    !---------------------------------------------------------------------------
    ! Free booster
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_free(booster)
        type(lgbm_booster_handle), intent(inout) :: booster
        integer(c_int) :: ret

        if (booster%is_valid()) then
            ret = LGBM_BoosterFree(booster%ptr)
            call lgbm_check_error(ret, "BoosterFree")
            booster%ptr = c_null_ptr
        end if
    end subroutine lgbm_booster_free

    !---------------------------------------------------------------------------
    ! Add validation data to booster
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_add_valid_data(booster, valid_data)
        type(lgbm_booster_handle), intent(in) :: booster
        type(lgbm_dataset_handle), intent(in) :: valid_data
        integer(c_int) :: ret

        ret = LGBM_BoosterAddValidData(booster%ptr, valid_data%ptr)
        call lgbm_check_error(ret, "BoosterAddValidData")
    end subroutine lgbm_booster_add_valid_data

    !---------------------------------------------------------------------------
    ! Update one iteration
    !---------------------------------------------------------------------------
    function lgbm_booster_update_one_iter(booster) result(is_finished)
        type(lgbm_booster_handle), intent(in) :: booster
        logical :: is_finished
        integer(c_int) :: ret, finished

        ret = LGBM_BoosterUpdateOneIter(booster%ptr, finished)
        call lgbm_check_error(ret, "BoosterUpdateOneIter")
        is_finished = (finished /= 0)
    end function lgbm_booster_update_one_iter

    !---------------------------------------------------------------------------
    ! Get evaluation results
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_get_eval(booster, data_idx, results)
        type(lgbm_booster_handle), intent(in) :: booster
        integer, intent(in) :: data_idx
        real(c_double), intent(out), allocatable :: results(:)

        integer(c_int) :: ret, out_len
        real(c_double), allocatable, target :: temp_results(:)

        ! Allocate temporary array (assume max 10 metrics)
        allocate(temp_results(10))

        ret = LGBM_BoosterGetEval(booster%ptr, int(data_idx, c_int), out_len, &
            c_loc(temp_results(1)))
        call lgbm_check_error(ret, "BoosterGetEval")

        ! Copy results
        allocate(results(out_len))
        results = temp_results(1:out_len)
        deallocate(temp_results)
    end subroutine lgbm_booster_get_eval

    !---------------------------------------------------------------------------
    ! Get current iteration
    !---------------------------------------------------------------------------
    function lgbm_booster_get_current_iteration(booster) result(iteration)
        type(lgbm_booster_handle), intent(in) :: booster
        integer :: iteration
        integer(c_int) :: ret, out_iter

        ret = LGBM_BoosterGetCurrentIteration(booster%ptr, out_iter)
        call lgbm_check_error(ret, "BoosterGetCurrentIteration")
        iteration = int(out_iter)
    end function lgbm_booster_get_current_iteration

    !---------------------------------------------------------------------------
    ! Predict for matrix (generic)
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_predict_for_mat(booster, data_ptr, data_type, nrow, ncol, &
            predict_type, num_iteration, predictions)
        type(lgbm_booster_handle), intent(in) :: booster
        type(c_ptr), intent(in) :: data_ptr
        integer(c_int), intent(in) :: data_type
        integer, intent(in) :: nrow, ncol
        integer, intent(in), optional :: predict_type, num_iteration
        real(c_double), intent(out), allocatable, target :: predictions(:)

        integer(c_int) :: ret, pred_type, start_iter, num_iter
        integer(c_int64_t) :: out_len
        character(len=1) :: empty_params

        pred_type = C_API_PREDICT_NORMAL
        start_iter = 0
        num_iter = -1

        if (present(predict_type)) pred_type = int(predict_type, c_int)
        if (present(num_iteration)) num_iter = int(num_iteration, c_int)

        ! Allocate predictions array
        allocate(predictions(nrow))
        empty_params = c_null_char

        ret = LGBM_BoosterPredictForMat(booster%ptr, data_ptr, data_type, &
            int(nrow, c_int32_t), int(ncol, c_int32_t), &
            0_c_int, pred_type, start_iter, num_iter, &  ! is_row_major=0
            empty_params, out_len, c_loc(predictions(1)))

        call lgbm_check_error(ret, "BoosterPredictForMat")
    end subroutine lgbm_booster_predict_for_mat

    !---------------------------------------------------------------------------
    ! Predict for double precision matrix
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_predict_for_mat_f64(booster, data, predictions, &
            predict_type, num_iteration)
        type(lgbm_booster_handle), intent(in) :: booster
        real(c_double), intent(in), target :: data(:,:)
        real(c_double), intent(out), allocatable :: predictions(:)
        integer, intent(in), optional :: predict_type, num_iteration

        integer :: nrow, ncol

        nrow = size(data, 1)
        ncol = size(data, 2)

        call lgbm_booster_predict_for_mat(booster, c_loc(data), C_API_DTYPE_FLOAT64, &
            nrow, ncol, predict_type, num_iteration, predictions)
    end subroutine lgbm_booster_predict_for_mat_f64

    !---------------------------------------------------------------------------
    ! Predict for single precision matrix
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_predict_for_mat_f32(booster, data, predictions, &
            predict_type, num_iteration)
        type(lgbm_booster_handle), intent(in) :: booster
        real(c_float), intent(in), target :: data(:,:)
        real(c_double), intent(out), allocatable :: predictions(:)
        integer, intent(in), optional :: predict_type, num_iteration

        integer :: nrow, ncol

        nrow = size(data, 1)
        ncol = size(data, 2)

        call lgbm_booster_predict_for_mat(booster, c_loc(data), C_API_DTYPE_FLOAT32, &
            nrow, ncol, predict_type, num_iteration, predictions)
    end subroutine lgbm_booster_predict_for_mat_f32

    !---------------------------------------------------------------------------
    ! Save model to file
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_save_model(booster, filename, num_iteration, &
            feature_importance_type)
        type(lgbm_booster_handle), intent(in) :: booster
        character(len=*), intent(in) :: filename
        integer, intent(in), optional :: num_iteration, feature_importance_type

        integer(c_int) :: ret, num_iter, feat_type
        character(len=:), allocatable :: c_filename

        num_iter = -1
        feat_type = C_API_FEATURE_IMPORTANCE_SPLIT

        if (present(num_iteration)) num_iter = int(num_iteration, c_int)
        if (present(feature_importance_type)) feat_type = int(feature_importance_type, c_int)

        c_filename = to_c_string(filename)

        ret = LGBM_BoosterSaveModel(booster%ptr, 0_c_int, num_iter, feat_type, c_filename)
        call lgbm_check_error(ret, "BoosterSaveModel")
    end subroutine lgbm_booster_save_model

    !---------------------------------------------------------------------------
    ! Dump model to string
    !---------------------------------------------------------------------------
    function lgbm_booster_dump_model(booster, num_iteration, feature_importance_type) &
            result(model_str)
        type(lgbm_booster_handle), intent(in) :: booster
        integer, intent(in), optional :: num_iteration, feature_importance_type
        character(len=:), allocatable :: model_str

        integer(c_int) :: ret, num_iter, feat_type
        integer(c_int64_t) :: buffer_len, out_len
        character(len=1), allocatable :: buffer(:)

        num_iter = -1
        feat_type = C_API_FEATURE_IMPORTANCE_SPLIT

        if (present(num_iteration)) num_iter = int(num_iteration, c_int)
        if (present(feature_importance_type)) feat_type = int(feature_importance_type, c_int)

        ! First call to get required buffer length
        buffer_len = 0
        allocate(buffer(1))
        ret = LGBM_BoosterDumpModel(booster%ptr, 0_c_int, num_iter, feat_type, &
            buffer_len, out_len, buffer)
        deallocate(buffer)

        ! Second call with proper buffer
        allocate(buffer(out_len + 1))
        ret = LGBM_BoosterDumpModel(booster%ptr, 0_c_int, num_iter, feat_type, &
            out_len + 1, out_len, buffer)
        call lgbm_check_error(ret, "BoosterDumpModel")

        ! Convert to Fortran string
        allocate(character(len=out_len) :: model_str)
        model_str = transfer(buffer(1:out_len), model_str)
        deallocate(buffer)
    end function lgbm_booster_dump_model

    !---------------------------------------------------------------------------
    ! Get feature importance
    !---------------------------------------------------------------------------
    subroutine lgbm_booster_feature_importance(booster, importance, num_iteration, &
            importance_type, n_features)
        type(lgbm_booster_handle), intent(in) :: booster
        real(c_double), intent(out), allocatable, target :: importance(:)
        integer, intent(in), optional :: num_iteration, importance_type
        integer, intent(in), optional :: n_features

        integer(c_int) :: ret, num_iter, imp_type
        integer :: nf

        num_iter = -1
        imp_type = C_API_FEATURE_IMPORTANCE_SPLIT
        nf = 100  ! safe default

        if (present(num_iteration)) num_iter = int(num_iteration, c_int)
        if (present(importance_type)) imp_type = int(importance_type, c_int)
        if (present(n_features)) nf = n_features

        allocate(importance(nf))

        ret = LGBM_BoosterFeatureImportance(booster%ptr, num_iter, imp_type, &
            c_loc(importance(1)))
        call lgbm_check_error(ret, "BoosterFeatureImportance")
    end subroutine lgbm_booster_feature_importance

end module lightgbm_api
