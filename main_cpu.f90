program test
use asyncio

call kernel1()

open(3,file='filename',form='unformatted',status='unknown')

call kernel2()

call asyncio_flush()

close(3)

end program test

