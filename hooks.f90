subroutine asyncio_open(unt, fname, frm, sta)
  implicit none
  integer :: unt
  character :: fname*(*)
  character :: frm*(*)
  character, optional :: sta*(*)
  if (present(sta)) then
    open(unt, file=fname, form=frm, status=sta)
  else
    open(unt, file=fname, form=frm)
  endif
end subroutine asyncio_open

subroutine asyncio_close(unt)
  implicit none
  integer :: unt
  close(unt)
end subroutine asyncio_close

subroutine asyncio_rewind(unt)
  implicit none
  integer :: unt
  rewind unt
end subroutine asyncio_rewind

subroutine asyncio_endfile(unt)
  implicit none
  integer ::unt
  endfile unt
end subroutine asyncio_endfile

subroutine asyncio_backspace(unt)
  implicit none
  integer ::unt
  endfile unt
end subroutine asyncio_backspace

subroutine asyncio_hook_write_default_unit_default_format(iost) bind(C)
  implicit none
  integer :: iost
  write(*,*,iostat=iost) 'hook_write'
end subroutine asyncio_hook_write_default_unit_default_format

subroutine asyncio_hook_write_default_unit_formatted(flen, frmt, iost) bind(C)
  use iso_c_binding
  implicit none
  integer(c_size_t), value :: flen
  character(c_char) :: frmt(flen)
  integer :: iost
  write(*,frmt,iostat=iost) 'hook_write'
end subroutine asyncio_hook_write_default_unit_formatted

subroutine asyncio_hook_write_unit_unformatted(unt,iost) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: unt
  integer :: iost
  write(unt,iostat=iost) 'hook_write'
end subroutine asyncio_hook_write_unit_unformatted

subroutine asyncio_hook_write_unit_default_format(unt, iost) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: unt
  integer :: iost
  write(unt,*,iostat=iost) 'hook_write'
end subroutine asyncio_hook_write_unit_default_format

subroutine asyncio_hook_write_unit_formatted(unt, flen, frmt, iost) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: unt
  integer(c_size_t), value :: flen
  character(c_char) :: frmt(flen)
  integer :: iost
  write(unt,frmt,iostat=iost) 'hook_write'
end subroutine asyncio_hook_write_unit_formatted

subroutine asyncio_hook_write_integer_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1
  integer(c_int) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_integer_array_1d

subroutine asyncio_hook_write_integer_array_2d(val, dim_1, dim_2) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1, dim_2
  integer(c_int) :: val(dim_1, dim_2)
  write(*,*) val
end subroutine asyncio_hook_write_integer_array_2d

subroutine asyncio_hook_write_float_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1
  real(c_float) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_float_array_1d

subroutine asyncio_hook_write_double_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1
  real(c_double) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_1d

subroutine asyncio_hook_write_double_array_2d(val, dim_1, dim_2) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1, dim_2
  real(c_double) :: val(dim_1, dim_2)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_2d

subroutine asyncio_hook_write_double_array_3d(val, dim_1, dim_2, dim_3) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1, dim_2, dim_3
  real(c_double) :: val(dim_1, dim_2, dim_3)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_3d

subroutine asyncio_hook_write_double_array_4d(val, dim_1, dim_2, dim_3, dim_4) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1, dim_2, dim_3, dim_4
  real(c_double) :: val(dim_1, dim_2, dim_3, dim_4)
  write(*,*) val
end subroutine asyncio_hook_write_double_array_4d

subroutine asyncio_hook_write_boolean_array_1d(val, dim_1) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: dim_1
  logical(c_bool) :: val(dim_1)
  write(*,*) val
end subroutine asyncio_hook_write_boolean_array_1d

