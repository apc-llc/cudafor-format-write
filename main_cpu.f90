program test
use asyncio

call kernel1()

call asyncio_open(3,file='filename',form='unformatted',status='unknown')

call kernel2()

call asyncio_flush()

call asyncio_close(3)

call kernel3()
call kernel3_cpu()

end program test

