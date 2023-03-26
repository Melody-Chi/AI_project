# 关于棋子的移动
# 给定一个棋子和方向，移动位置并且更新棋盘
from .utils import render_board


# def move(board: dict[tuple, tuple], ansi=False) -> 
def random_move(input, direction=None):
    for key in input.keys():
        r =key[0]
        q = key[1]
        print(r)
        if direction == 'up':
            if q % 2 == 0:
                r -= 1
            q -= 1
        elif direction == 'down':
            if q % 2 == 0:
                r += 1
            q += 1
        elif direction == 'upright':
            r -= 1
        elif direction == 'downright':
            r += 1
        elif direction == 'upleft':
            r -= 1
            q -= 1
        elif direction == 'downleft':
            r += 1
            q -= 1
        else:
            return (r,q)
        # Check if new position is within the board boundaries
        if r < 0:
            r = r + 7
        elif r > 6:
            r = r -7
        if q < 0:
            q = q+7
        elif q > 6:
            q = q-7

        return (r, q)
