# Recipe from Python Cookbook, 2.23 Reading an Unbuffered Character in a Cross-Platform Way
import sys

try:
    from msvcrt import getwch as getch
except ImportError:
    """We are not on Windows; try the Unix-like approach"""

    def getch():
        import sys, tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


SKIP = False
STEPS_TO_SKIP = 0


def wait():
    global SKIP, STEPS_TO_SKIP
    if SKIP:
        return
    if STEPS_TO_SKIP > 0:
        STEPS_TO_SKIP -= 1
        return
    print("> [N]ext step, skip [<num>] steps, [r]un, [q]uit: ", end="")
    sys.stdout.flush()
    while True:
        key = getch().lower()
        # Remove the prompt from output
        if key not in "123456789nrqNRQ":
            continue
        print(
            "\r                                                             \r", end=""
        )
        if key == "n":
            break
        if key.isdigit():
            STEPS_TO_SKIP = int(key)
            break
        if key == "r":  # run
            SKIP = True
            break
        if key == "q":
            sys.exit(1)
