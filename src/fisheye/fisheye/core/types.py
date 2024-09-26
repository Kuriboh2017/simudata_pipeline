from enum import Enum


def _float_hash(*args):
    # Flatten the args into a list of numbers
    numbers = [num for arg in args for num in (
        arg if isinstance(arg, (list, tuple)) else (arg,))]
    hashes = [hash(f"{num:.6f}") for num in numbers]
    return hash(tuple(hashes))


class CameraType(Enum):
    PANORAMA = "Panorama"
    FISHEYE = "Fisheye"
    THREE_PINHOLES = "ThreePinholes"


class Intrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.items = [fx, fy, cx, cy]

    def __str__(self):
        return (f'Intrinsics(fx={self.fx}, fy={self.fy}, cx={self.cx}, '
                f'cy={self.cy}) ')

    def __hash__(self) -> int:
        return _float_hash((self.fx, self.fy, self.cx, self.cy))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other) if isinstance(other, Intrinsics) else False

    def __getitem__(self, index):
        return self.items[index]

    @classmethod
    def from_array(cls, arr):
        assert len(arr) == 4, "Array must have 4 elements."
        return cls(*arr)


class EucmIntrinsics(Intrinsics):
    def __init__(self, fx, fy, cx, cy, alpha, beta) -> None:
        super().__init__(fx, fy, cx, cy)  # Use super() to call the __init__ of Intrinsics
        self.alpha = alpha
        self.beta = beta
        self.items.extend([alpha, beta])  # Add alpha and beta to items

    def __str__(self):
        return (f'EucmIntrinsics(intrinsics={super().__str__()},'
                f'alpha={self.alpha}, beta={self.beta}) ')

    def __hash__(self) -> int:
        return hash((super().__hash__(), _float_hash(self.alpha), _float_hash(self.beta)))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other) if isinstance(other, EucmIntrinsics) else False

    @classmethod
    def from_array(cls, arr):
        assert len(arr) == 6, "Array must have 6 elements."
        return cls(*arr)


class Rotation:
    def __init__(self, r, p, y):
        self.r = r
        self.p = p
        self.y = y
        self.items = [r, p, y]

    def __hash__(self) -> int:
        return _float_hash((self.r, self.p, self.y))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other) if isinstance(other, Rotation) else False

    def __getitem__(self, index):
        return self.items[index]


class ImageParams:
    def __init__(self, camera_type: CameraType, width: int, height: int, intrinsics: Intrinsics = None):
        self.camera_type = camera_type
        self.width = width
        self.height = height
        self.intrinsics = intrinsics

    def __hash__(self) -> int:
        return hash((_float_hash(self.width), _float_hash(self.height), hash(self.intrinsics)))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other) if isinstance(other, ImageParams) else False


class RemappingTableParams:
    def __init__(self,  src_image_params: ImageParams, dst_image_params: ImageParams, rpy: Rotation):
        self.src = src_image_params
        self.dst = dst_image_params
        self.rpy = rpy

    def __hash__(self) -> int:
        return hash((self.src, self.dst, self.rpy))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other) if isinstance(other, RemappingTableParams) else False


class RemappingTable:
    def __init__(self, down_x, down_y, up_x=None, up_y=None):
        self.down_x = down_x
        self.down_y = down_y
        self.up_x = up_x
        self.up_y = up_y

    def __iter__(self):
        return iter((self.down_x, self.down_y, self.up_x, self.up_y))
