#include <unistd.h>

#ifdef MODULE_TEST_SCOREP_FINALIZE
/* Declared in Score-P's internal SCOREP_RuntimeManagement.h. */
extern void SCOREP_FinalizeMeasurement(void);
#endif

void module_test_exit_now(int status)
{
#ifdef MODULE_TEST_SCOREP_FINALIZE
    SCOREP_FinalizeMeasurement();
#endif
    _exit(status);
}
