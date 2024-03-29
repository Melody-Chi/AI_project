# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from dataclasses import dataclass

from .hex import HexPos, HexDir


# Here we define dataclasses for the two possible actions that a player can
# make. See the `hex.py` file for the definition of the `HexPos` and `HexDir`.
# If you are unfamiliar with dataclasses, see the relevant Python docs here:
# https://docs.python.org/3/library/dataclasses.html 


# frozen=True: 这使得数据类的实例是不可改变的。一旦一个实例被创建，它的属性就不能被改变。
# slots=True: 这告诉数据类使用 __slots__ 来存储它的实例属性，而不是使用一个字典。
# 这可以节省内存并加速属性的访问，但代价是一些灵活性，例如不允许动态地添加额外的属性。
@dataclass(frozen=True, slots=True)
class SpawnAction():
    cell: HexPos

    def __str__(self) -> str:
        return f"SPAWN({self.cell.r}, {self.cell.q})"


@dataclass(frozen=True, slots=True)
class SpreadAction():
    cell: HexPos
    direction: HexDir

    def __str__(self) -> str:
        return f"SPREAD({self.cell.r}, {self.cell.q}, " + \
               f"{self.direction.r}, {self.direction.q})"


Action = SpawnAction | SpreadAction
