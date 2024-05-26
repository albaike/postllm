extern "C" {
    #include "postgres.h"
    #include "fmgr.h"
    #include "miscadmin.h"
    #include "mb/pg_wchar.h"
    #include "utils/builtins.h"
    #include "storage/ipc.h"
    #include "storage/shmem.h"
    #include "storage/lwlock.h"
    #include "utils/hsearch.h"
    #include "catalog/pg_type.h"
}