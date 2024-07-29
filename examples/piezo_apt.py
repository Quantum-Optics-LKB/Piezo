import thorlabs_apt_device as apt
from thorlabs_apt_device.enums import EndPoint


class LTS(apt.APTDevice_Motor):
    """A class for the Long Travel Stage (LTS) device.
    Inherits from the APTDevice_Motor class.
    """

    def __init__(
        self,
        serial_port=None,
        vid=None,
        pid=None,
        manufacturer=None,
        product=None,
        serial_number=None,
        location=None,
        home=True,
        invert_direction_logic=False,
        swap_limit_switches=False,
        status_updates="none",
        controller=EndPoint.RACK,
        bays=(EndPoint.BAY0,),
        channels=(1,),
    ):
        super().__init__(
            serial_port,
            vid,
            pid,
            manufacturer,
            product,
            serial_number,
            location,
            home,
            invert_direction_logic,
            swap_limit_switches,
            status_updates,
            controller,
            bays,
            channels,
        )

    def _process_message(self, m):
        super()._process_message(m)

    def move_absolute(self, position=None, now=True, bay=0, channel=0):
        target_pos = position
        super().move_absolute(position, now, bay, channel)
        while self.status_[bay][channel]["position"] != target_pos:
            time.sleep(0.1)

    def home(self, bay: int = 0, channel: int = 0):
        super().home(bay, channel)
        time.sleep(0.05)
        is_homed = self.status_[bay][channel]["homing"]
        is_homed &= not self.status_[bay][channel]["homed"]
        is_moving = self.status_[bay][channel]["moving_forward"]
        is_moving |= self.status_[bay][channel]["moving_backward"]
        while not is_homed and is_moving:
            time.sleep(0.1)


if __name__ == "__main__":
    import time
    from functools import partial

    # Build our custom coversions using mm, mm/s and mm/s/s
    from_mm = partial(apt.from_pos, factor=34304)
    from_mmps = partial(apt.from_vel, factor=34304, t=2048 / 6e6)
    from_mmpsps = partial(apt.from_acc, factor=34304, t=2048 / 6e6)
    to_mm = partial(apt.from_pos, factor=1 / 34304)
    to_mmps = partial(apt.from_vel, factor=1 / 34304, t=2048 / 6e6)
    to_mmpsps = partial(apt.from_acc, factor=1 / 34304, t=2048 / 6e6)
    stage = LTS(serial_number="45832743", status_updates="polled")
    stage.identify()
    stage.set_velocity_params(100000, 150000)
    print(stage.velparams_)
    print(stage.status_)
    stage.home()
    print(stage.status_)
    stage.close()
