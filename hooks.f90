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
  write(unt,*) 'hook_write'
end subroutine asyncio_hook_write_unit_unformatted

subroutine asyncio_hook_write_unit_formatted(unt, flen, frmt) bind(C)
  use iso_c_binding
  integer(c_int), value :: unt
  integer(c_size_t), value :: flen
  character(c_char) :: frmt(flen)
  write(unt,frmt) 'hook_write'
end subroutine asyncio_hook_write_unit_formatted

