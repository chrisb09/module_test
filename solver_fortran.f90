! =============================================================================
! Fortran API solver driver for the module_test matrix.
!
! Mirrors solver.cpp for API_MODE=STATIC using the Fortran cmi module:
!   config file -> create coupling with live buffers + model_file override
!   -> step -> MPI gather -> print the machine-readable output blocks.
! =============================================================================

program module_test_solver_fortran
  use, intrinsic :: iso_c_binding
  use mpi
  use cmi
  implicit none

  character(len=4096) :: config_path, provider, model_name, api_mode
  character(len=4096) :: merge_strategy, base_dir, actual_model_path
  character(len=256) :: env_buf
  character(kind=c_char), allocatable, target :: config_c(:), model_c(:)
  character(len=:), allocatable :: line

  integer :: ierr, world_rank, world_size, local_comm, local_size
  integer :: batch_size, in_size, out_size, in_ndims, out_ndims
  integer :: env_len, env_status
  integer :: is_mmcp, steps, step, rank, step_idx, i, b, offset
  integer :: merge_strategy_value

  real(c_float), allocatable, target :: flat_data(:)
  real(c_float), allocatable, target :: output_buffer(:)
  real(c_float), allocatable :: outputs_first(:), outputs_sum(:)
  real(c_float), allocatable :: all_first(:, :), all_sums(:, :)
  real(c_float) :: checksum

  integer(c_int) :: in_dims(3), out_dims(3)
  integer(c_int) :: st, delta
  type(cmi_tensor) :: input_tensor, output_tensor
  type(cmi_data) :: input_data, output_data
  type(cmi_config_overrides) :: overrides
  type(cmi_coupling) :: coupling

  interface
     ! Skip static teardown, which can hang for provider runtime workers.
     subroutine c_exit_now(status) bind(C, name="_exit")
       import :: c_int
       integer(c_int), value :: status
     end subroutine c_exit_now
  end interface

  call get_command_argument(1, config_path)
  if (len_trim(config_path) == 0) config_path = "config.toml"

  call get_environment_variable("PROVIDER", provider)
  if (len_trim(provider) == 0) then
     write(*, '(a)') "[module_test_fortran] PROVIDER is not set. Aborting."
     stop 1
  end if

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, world_rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, world_size, ierr)

  ! Match the C++ solver's hardcoded solver color.
  call MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, local_comm, ierr)
  if (local_comm == MPI_COMM_NULL) then
     if (world_rank == 0) then
        write(*, '(a)') "[module_test_fortran] Solver communicator is null; check MPMD split logic."
     end if
     call MPI_Finalize(ierr)
     stop 1
  end if
  call MPI_Comm_size(local_comm, local_size, ierr)

  write(*, '(a,i0,a)') "Hello from rank ", world_rank, "!"
  if (world_rank == 0) then
     write(*, '(a,a,a)') "Running with ", trim(provider), " provider"
     write(*, '(a,a)') "Loading config from ", trim(config_path)
  end if
  flush(6)

  call get_environment_variable("API_MODE", api_mode)
  if (len_trim(api_mode) == 0) api_mode = "STATIC"
  if (trim(api_mode) /= "STATIC") then
     write(*, '(a)') "[module_test_fortran] Fortran solver currently supports only API_MODE=STATIC."
     call MPI_Finalize(ierr)
     stop 1
  end if

  call get_environment_variable("MODEL", model_name)
  is_mmcp = 0
  if (index(model_name, "mmcp") > 0) is_mmcp = 1

  call get_environment_variable("BATCH_SIZE", env_buf, env_len, env_status)
  batch_size = 1
  if (env_status == 0 .and. env_len > 0) read(env_buf(1:env_len), *) batch_size

  if (is_mmcp == 1) then
     in_size = 5 * 512 * batch_size
     out_size = 2 * 512 * batch_size
     in_dims = [int(batch_size, c_int), 5_c_int, 512_c_int]
     out_dims = [int(batch_size, c_int), 2_c_int, 512_c_int]
     in_ndims = 3
     out_ndims = 3
  else
     in_size = 18 * batch_size
     out_size = 1 * batch_size
     in_dims = [int(batch_size, c_int), 18_c_int, 0_c_int]
     out_dims = [int(batch_size, c_int), 0_c_int, 0_c_int]
     in_ndims = 2
     out_ndims = 1
  end if

  allocate(flat_data(in_size))
  allocate(output_buffer(out_size))

  if (is_mmcp == 1) then
     do i = 1, in_size
        flat_data(i) = real(mod(4 + (i - 1) * 17, 100), c_float)
     end do
  else
     do b = 0, batch_size - 1
        offset = b * 18
        do i = 0, 8
           flat_data(offset + i + 1) = real(mod(4 + i * 17, 100), c_float)
        end do
        do i = 0, 8
           flat_data(offset + 9 + i + 1) = real(mod(7 + i * 24, 200), c_float)
        end do
     end do
  end if
  do i = 1, in_size
     flat_data(i) = flat_data(i) * real(world_rank, c_float)
  end do
  output_buffer = -1.0_c_float

  call to_c_string(config_path, config_c)

  st = c_cmi_tensor_create_flat(input_tensor%ptr, c_loc(flat_data), in_dims, int(in_ndims, c_int), &
                                CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL)
  call require_success(st, "create input tensor")
  st = c_cmi_data_create(input_data%ptr, CMI_DTYPE_FLOAT)
  call require_success(st, "create input data")
  st = c_cmi_data_add_tensor(input_data%ptr, input_tensor%ptr)
  call require_success(st, "add input tensor")

  st = c_cmi_tensor_create_flat(output_tensor%ptr, c_loc(output_buffer), out_dims, int(out_ndims, c_int), &
                                CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS, CMI_OWNERSHIP_EXTERNAL)
  call require_success(st, "create output tensor")
  st = c_cmi_data_create(output_data%ptr, CMI_DTYPE_FLOAT)
  call require_success(st, "create output data")
  st = c_cmi_data_add_tensor(output_data%ptr, output_tensor%ptr)
  call require_success(st, "add output tensor")

  if (is_mmcp == 1) then
     actual_model_path = "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/" // &
                          "input/transformer_inference_scripted_fw2.pt"
  else
     call get_environment_variable("BASE_DIR", base_dir)
     if (len_trim(base_dir) == 0) base_dir = "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
     actual_model_path = trim(base_dir) // "/mini_app/train_models/model_a/" // trim(model_name) // "_cpu.pt"
  end if

  st = c_cmi_config_overrides_create(overrides%ptr)
  call require_success(st, "create config overrides")
  if (trim(provider) /= "SMARTSIM") then
     call to_c_string(actual_model_path, model_c)
     st = c_cmi_config_overrides_set_dotted(overrides%ptr, "library.model_file"//c_null_char, &
                                            CMI_PARAM_STRING, c_loc(model_c))
     call require_success(st, "set model_file override")
  end if

  call get_environment_variable("MERGE_STRATEGY", merge_strategy)
  merge_strategy_value = -1
  select case (trim(merge_strategy))
  case ("STACK")
     merge_strategy_value = CMI_MERGE_STACK
  case ("LIST")
     merge_strategy_value = CMI_MERGE_LIST
  case ("AUTO")
     merge_strategy_value = CMI_MERGE_AUTO
  case ("NONE")
     merge_strategy_value = CMI_MERGE_NONE
  end select

  st = c_cmi_coupling_create_from_config_with_data_and_overrides( &
       coupling%ptr, config_c, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, &
       CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, input_data%ptr, output_data%ptr, overrides%ptr)
  call require_success(st, "create coupling from config")
  if (.not. c_associated(coupling%ptr)) then
     write(*, '(a)') "[module_test_fortran] Coupling handle is null."
     call MPI_Finalize(ierr)
     stop 2
  end if

  if (merge_strategy_value >= 0) then
     st = c_cmi_coupling_set_merge_strategy(coupling%ptr, int(merge_strategy_value, c_int))
     call require_success(st, "set merge strategy")
  end if

  call get_environment_variable("STEPS", env_buf, env_len, env_status)
  steps = 1
  if (env_status == 0 .and. env_len > 0) read(env_buf(1:env_len), *) steps
  if (steps < 1) steps = 1

  allocate(outputs_first(steps))
  allocate(outputs_sum(steps))

  do step = 0, steps - 1
     do i = 1, in_size
        flat_data(i) = flat_data(i) + real(step, c_float)
     end do

     delta = 0_c_int
     st = c_cmi_coupling_step(coupling%ptr, delta)
     if (st /= CMI_SUCCESS) then
        write(*, '(a,i0)') "[module_test_fortran] Inference failed at step ", step
        call MPI_Finalize(ierr)
        stop 3
     end if

     if (world_rank == 0) then
        write(*, '(a,f0.6,a)') "Inference output: [", output_buffer(1), "]"
     end if
     outputs_first(step + 1) = output_buffer(1)

     checksum = 0.0_c_float
     do i = 1, out_size
        checksum = checksum + output_buffer(i)
     end do
     outputs_sum(step + 1) = checksum
  end do

  allocate(all_first(steps, local_size))
  allocate(all_sums(steps, local_size))
  call MPI_Gather(outputs_first, steps, MPI_REAL4, all_first, steps, MPI_REAL4, 0, local_comm, ierr)
  call MPI_Gather(outputs_sum, steps, MPI_REAL4, all_sums, steps, MPI_REAL4, 0, local_comm, ierr)

  if (world_rank == 0) then
     write(*, '(a)') ""
     write(*, '(a)') "###########################################################################"
     write(*, '(a)') "Gathered output[0] from all ranks:"
     line = "[["
     do rank = 1, local_size
        if (rank > 1) line = line // "], ["
        do step_idx = 1, steps
           if (step_idx > 1) line = line // ", "
           call append_real(line, all_first(step_idx, rank))
        end do
     end do
     line = line // "]]"
     write(*, '(a)') line

     write(*, '(a)') "Gathered output sums from all ranks:"
     line = "[["
     do rank = 1, local_size
        if (rank > 1) line = line // "], ["
        do step_idx = 1, steps
           if (step_idx > 1) line = line // ", "
           call append_real(line, all_sums(step_idx, rank))
        end do
     end do
     line = line // "]]"
     write(*, '(a)') line
     flush(6)
  end if

  ! Match the C++/C drivers: destroy the coupling (provider finalize) first.
  st = c_cmi_coupling_destroy(coupling%ptr)
  st = c_cmi_config_overrides_destroy(overrides%ptr)
  st = c_cmi_data_destroy(input_data%ptr)
  st = c_cmi_data_destroy(output_data%ptr)
  st = c_cmi_tensor_destroy(input_tensor%ptr)
  st = c_cmi_tensor_destroy(output_tensor%ptr)

  call MPI_Finalize(ierr)
  call c_exit_now(0_c_int)

contains

  subroutine to_c_string(src, dst)
    character(len=*), intent(in) :: src
    character(kind=c_char), allocatable, target, intent(inout) :: dst(:)
    integer :: n, idx

    if (allocated(dst)) deallocate(dst)
    n = len_trim(src)
    allocate(dst(n + 1))
    do idx = 1, n
       dst(idx) = src(idx:idx)
    end do
    dst(n + 1) = c_null_char
  end subroutine to_c_string

  subroutine append_real(target_line, value)
    character(len=:), allocatable, intent(inout) :: target_line
    real(c_float), intent(in) :: value
    character(len=32) :: tmp

    write(tmp, '(f8.4)') value
    target_line = target_line // trim(adjustl(tmp))
  end subroutine append_real

  subroutine require_success(status, what)
    integer(c_int), intent(in) :: status
    character(len=*), intent(in) :: what

    if (status /= CMI_SUCCESS) then
       write(*, '(a,a,a)') "[module_test_fortran] ", trim(what), " failed."
       call MPI_Finalize(ierr)
       stop 1
    end if
  end subroutine require_success

end program module_test_solver_fortran
