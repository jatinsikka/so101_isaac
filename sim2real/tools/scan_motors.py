import sys
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS

BAUDRATES = [1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400, 19_200]
SO101_JOINTS = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}

def main(port_name):
    port = PortHandler(port_name)
    if not port.openPort():
        print(f"Could not open {port_name}. Is the arm plugged in and powered?")
        return

    packet = PacketHandler(0)
    found_any = False

    for baud in BAUDRATES:
        port.setBaudRate(baud)
        found = []
        for scs_id in range(1, 21):
            model, comm, err = packet.ping(port, scs_id)
            if comm == COMM_SUCCESS:
                found.append((scs_id, model, err))

        if found:
            found_any = True
            print(f"\nbaudrate {baud}:")
            for scs_id, model, err in found:
                joint = SO101_JOINTS.get(scs_id, "")
                note = f"  <- expected: {joint}" if joint else "  <- unexpected ID"
                errtxt = f"  (error flag {err})" if err else ""
                print(f"  ID {scs_id:>2}  model {model}{errtxt}{note}")

    if not found_any:
        print("\nNothing responded at any baudrate.")
        print("Check: 12V power connected and switched on, servo bus cable seated,")
        print("and that this is the arm's port rather than the other arm's.")

    port.closePort()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem5AAF2879831")
