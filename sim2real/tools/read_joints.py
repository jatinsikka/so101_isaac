import sys, time
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

ADDR_PRESENT_POSITION = 56   # STS3215
ADDR_PRESENT_SPEED    = 58
JOINTS = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
TICKS_PER_REV = 4096         # assumption to verify

def main(port_name):
    port = PortHandler(port_name)
    if not port.openPort():
        print(f"Could not open {port_name}"); return
    port.setBaudRate(1_000_000)
    packet = PacketHandler(0)

    seen = {i: [] for i in range(1, 7)}
    print("Move the arm by hand. Ctrl-C when done.\n")
    print("      " + "".join(f"{j[:12]:>14}" for j in JOINTS))
    try:
        while True:
            ticks = []
            for scs_id in range(1, 7):
                pos, comm, err = packet.read2ByteTxRx(port, scs_id, ADDR_PRESENT_POSITION)
                ticks.append(pos if comm == COMM_SUCCESS else None)
                if comm == COMM_SUCCESS:
                    seen[scs_id].append(pos)
            print("tick  " + "".join(f"{t if t is not None else '--':>14}" for t in ticks))
            print("deg   " + "".join(
                f"{(t*360.0/TICKS_PER_REV):>14.1f}" if t is not None else f"{'--':>14}" for t in ticks))
            print()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n=== range observed per joint (ticks) ===")
        for scs_id, vals in seen.items():
            if vals:
                print(f"  ID {scs_id} {JOINTS[scs_id-1]:<15} min {min(vals):>5}  max {max(vals):>5}  last {vals[-1]:>5}")
    port.closePort()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem5AAF2879831")
