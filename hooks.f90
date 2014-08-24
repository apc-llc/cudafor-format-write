subroutine asyncio_hook_write_default_unformatted() bind(C)
  write(*,*) 'hook_write'
end subroutine asyncio_hook_write_default_unformatted

subroutine asyncio_hook_write_default_formatted(flen, frmt) bind(C)
  use iso_c_binding
  integer(c_size_t), value :: flen
  character(c_char) :: frmt(flen)
  write(*,frmt) 'hook_write'
end subroutine asyncio_hook_write_default_formatted

subroutine asyncio_hook_write_unit_unformatted(unt) bind(C)
  use iso_c_binding
  integer(c_int), value :: unt
  write(unt) 'hook_write'
end subroutine asyncio_hook_write_unit_unformatted

subroutine asyncio_hook_write_unit_formatted(unt, flen, frmt) bind(C)
  use iso_c_binding
  integer(c_int), value :: unt
  integer(c_size_t), value :: flen
  character(c_char) :: frmt(flen)
  write(unt,frmt) 'hook_write'
end subroutine asyncio_hook_write_unit_formatted

subroutine asyncio_hook_write_integer_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1
  integer(c_int) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_integer_array_1d

subroutine asyncio_hook_write_integer_array_2d(val, dim_1, dim_2) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1, dim_2
  integer(c_int) :: val(dim_1, dim_2)
  write(*,*) val
end subroutine asyncio_hook_write_integer_array_2d

subroutine asyncio_hook_write_float_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1
  real(c_float) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_float_array_1d

subroutine asyncio_hook_write_double_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1
  real(c_double) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_1d

subroutine asyncio_hook_write_double_array_2d(val, dim_1, dim_2) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1, dim_2
  real(c_double) :: val(dim_1, dim_2)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_2d

subroutine asyncio_hook_write_double_array_3d(val, dim_1, dim_2, dim_3) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1, dim_2, dim_3
  real(c_double) :: val(dim_1, dim_2, dim_3)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_3d

subroutine asyncio_hook_write_double_array_4d(val, dim_1, dim_2, dim_3, dim_4) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1, dim_2, dim_3, dim_4
  real(c_double) :: val(dim_1, dim_2, dim_3, dim_4)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_4d

subroutine asyncio_hook_write_boolean_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  integer(c_int), value :: dim_1
  logical(c_bool) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_boolean_array_1d

