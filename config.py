# spec06-int * 2, fp *2
# case_name = "403.2-ref-1"
# case_name = "437.1-ref-10"
# case_name = "471.1-ref-1"
# case_name = "453.1-ref-16"

# spec17-int * 9
#case_name = "500.1-refrate-1"
#case_name = "502.1-refrate-1" ### not ready
#case_name = "505.1-refrate-1"
#case_name = "523.1-refrate-1"
#case_name = "525.1-refrate-1"
#case_name = "531.1-refrate-1"
#case_name = "541.1-refrate-1"
# case_name = "548.1-refrate-1"
# case_name = "557.1-refrate-1"
# spec17-fp * 13
# case_name = "503.1-refrate-1"
# case_name = "507.1-refrate-1"
# case_name = "508.1-refrate-1"
# case_name = "510.1-refrate-1"
# case_name = "511.1-refrate-1"
# case_name = "519.1-refrate-1"
# case_name = "521.1-refrate-1"
# case_name = "526.1-refrate-1"
# case_name = "527.1-refrate-1"
# case_name = "538.1-refrate-1"
# case_name = "544.1-refrate-1"
# case_name = "549.1-refrate-1"
# case_name = "554.1-refrate-1"
import socket
import sys

if 1 < len(sys.argv):
    if '502' == sys.argv[1]:
        case_name = sys.argv[1] + ".2-refrate-1"
    else:
        case_name = sys.argv[1] + ".1-refrate-1"
else:
    #case_name = "500.1-refrate-1"
    #case_name = "519.1-refrate-2"
    case_name = "500.1-refrate-1"


smoke_test = False

mape_line_analysis = True
plot_pareto = False
hostname = socket.getfqdn(socket.gethostname())

if "SC-202005121725" == hostname:
    plot_pareto = False

if 3 < len(sys.argv):
    exp_id = int(sys.argv[3])
    mape_line_analysis = True
else:
    exp_id = None

N_SAMPLES_ALL = 4 if smoke_test else 80
N_SAMPLES_INIT = 2 if smoke_test else 25