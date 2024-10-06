import copy
import datetime
import math
import sys
import pynput
import pyautogui
import time
from PIL import Image
from enum import Enum
from typing import Literal

class MinesweeperException(Exception):
    def __init__(self, message):
        super().__init__(message)

class Color:
    def __init__(self, hex_code: int):
        self.hex_code = hex_code
        
        h = hex_code
        self.b = h % 256
        h = (h - self.b) / 256
        self.g = h % 256
        h = (h - self.g) / 256
        self.r = h % 256
    
    @staticmethod
    def from_rgb(rgb: tuple[int, int, int]):
        hex = rgb[0]
        hex = rgb[1] + 256*hex
        hex = rgb[2] + 256*hex
        return Color(hex)
    
    @staticmethod
    def from_pixel(image: Image.Image, x: int, y: int):
        return Color.from_rgb(image.getpixel([x, y]))
    
    def __repr__(self):
        return f"Color#{hex(self.hex_code)[2:].upper()}"
    
    def __get_quantitative_diff(self, other) -> int:
        c1 = [self.r, self.g, self.b]
        c2 = [other.r, other.g, other.b]
        diffs = [ (c1[i]-c2[i])**2 for i in range(3) ]
        return diffs[0] + diffs[1] + diffs[2]
    
    def compare(self, other, tol: int) -> bool:
        return self.__get_quantitative_diff(other) <= tol
    
    def get_best_match(self, other_colors) -> int:
        """
        Takes an `other_colors` list of `Color` objects.
        Returns the index of the best match to this color in
        the list
        """
        idx_diffs = [
            [idx, self.__get_quantitative_diff(other)]
            for idx, other in enumerate(other_colors)
        ]
        idx_diffs.sort(key=lambda idx_diff: (idx_diff[1], idx_diff[0]))
        return idx_diffs[0][0]
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Color):
            return self.compare(other, 0)
        else:
            return NotImplemented

class Vec2D:
    def __init__(self, x: float|int, y: float|int):
        self.x = float(x)
        self.y = float(y)
    
    def __mul__(self, other):
        if isinstance(other, float):
            return Vec2D(self.x * other, self.y * other)
        else:
            return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, float):
            return Vec2D(self.x / other, self.y / other)
        else:
            return NotImplemented
    
    def __add__(self, other):
        if isinstance(other, Vec2D):
            return Vec2D(self.x + other.x, self.y + other.y)
        else:
            return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Vec2D):
            return Vec2D(self.x - other.x, self.y - other.y)
        else:
            return NotImplemented
    
    def __neg__(self):
        return Vec2D(-self.x, -self.y)
    
    def __pos__(self):
        return self
    
    def __repr__(self):
        return f"<{self.x}, {self.y}>"
    
    def multiply_components(self, vec_or_x, y=None):
        """
        `multiply_components(other_vec)` works, as does `multiply_components(some_x_float, some_y_float)`
        
        (Hadamard product)
        """
        if isinstance(vec_or_x, Vec2D):
            return Vec2D(self.x * vec_or_x.x, self.y * vec_or_x.y)
        elif isinstance(vec_or_x, float):
            if isinstance(y, float):
                return Vec2D(self.x * vec_or_x, self.y * y)
            else:
                raise TypeError("y is not float value")
        else:
            raise TypeError("vec_or_x is neither Vec2D nor float")
    
    def interpolate(self, other, k: float):
        """
        Linear interpolation between this vector and the `other`, by
        `k`, such that a `k` of 0 yields a vector equivalent to this
        one and a `k` of 1 yields a vector equivalent to `other.
        """
        return (other - self) * k + self
    
    def components(self) -> tuple[float, float]:
        return [self.x, self.y]

class Resources(Enum):
    FLAG_ICON = "resources/flag_icon.png"
    
    COLOR_GREEN_TILE_LIGHT = Color(0xA9D652)
    COLOR_GREEN_TILE_DARK = Color(0xA1D04A)
    
    COLOR_MINES_1 = Color(0x1976D2)
    COLOR_MINES_2 = Color(0x388E3C)
    COLOR_MINES_3 = Color(0xD32F2F)
    COLOR_MINES_4 = Color(0x7B1FA2)
    COLOR_MINES_5 = Color(0xFF8F00)
    COLOR_MINES_6 = Color(0x0097A7)
    COLOR_MINES_7 = Color(0x424242)
    COLOR_MINES_8 = Color(0x9E9C9A)

class GridString:
    def __init__(self):
        self.__lines: list[str] = []
        self.__x_start = 0
        self.__y_start = 0
    
    def get_full_string(self) -> str:
        return "\n".join(self.__lines)+"\n"
    
    def __repr__(self):
        return self.get_full_string()
    
    def put_char(self, char: str, x: int, y: int):
        char = char[0]
        
        # Shift everything to the right if need be
        if x < self.__x_start:
            right_shift_by = self.__x_start - x
            for idx, line in enumerate(self.__lines):
                self.__lines[idx] = " " * right_shift_by + line
            self.__x_start = x
        
        # Shift everything down if need be
        if y < self.__y_start:
            down_shift_by = self.__y_start - y
            self.__lines = [ "" for _ in range(down_shift_by) ] + self.__lines
            self.__y_start = y
        
        # Get the true x and y indices based on where the grid starts
        true_x_idx = x - self.__x_start
        true_y_idx = y - self.__y_start
        
        # Create extra lines down if need be
        if true_y_idx >= len(self.__lines):
            new_lines = 1 + true_y_idx - len(self.__lines)
            self.__lines = self.__lines + [ "" for _ in range(new_lines) ]
        
        # Create extra spaces within correct line if need be
        if true_x_idx >= len(self.__lines[true_y_idx]):
            new_spaces = 1 + true_x_idx - len(self.__lines[true_y_idx])
            self.__lines[true_y_idx] = self.__lines[true_y_idx] + " " * new_spaces
        
        # Modify the line by inserting the character
        new_line = self.__lines[true_y_idx]
        new_line = new_line[:true_x_idx] + char + new_line[(true_x_idx+1):]
        self.__lines[true_y_idx] = new_line
    
    def get_char(self, x: int, y: int) -> str:
        true_x_idx = x - self.__x_start
        true_y_idx = y - self.__y_start
        if true_y_idx < 0 or true_y_idx >= len(self.__lines):
            return " "
        if true_x_idx < 0 or true_x_idx >= len(self.__lines[true_y_idx]):
            return " "
        return self.__lines[true_y_idx][true_x_idx]
    
    def put_str(self, content, x: int, y: int, dir:Literal["down"]|Literal["right"]="right", justify:Literal["start"]|Literal["center"]|Literal["end"]="start"):
        content = str(content)
        char_xy = [x, y]
        
        # Get offsets for different justifications
        offset = 0 # 0 for "start"
        if justify == "center":
            offset = math.ceil(len(content) / 2) - 1
        elif justify == "end":
            offset = len(content) - 1
        
        # Adjust for offset from justification
        comp_idx = 1 if (dir == "down") else 0
        char_xy[comp_idx] -= offset
        
        for char in content:
            self.put_char(char, char_xy[0], char_xy[1])
            char_xy[comp_idx] += 1

class GameGridCalibration:
    
    flag_icon_position: Vec2D
    
    edge_coords = tuple[Vec2D, Vec2D]
    """
    Top left xy and bottom right xy
    """
    
    num_tile_cols: int
    num_tile_rows: int
    
    def __init__(self, debug: bool):
        self.debug = debug
        self.calibrate_grid()
    
    def get_position(self, col: int, row: int, position_in_tile: Vec2D = Vec2D(0.5, 0.5)) -> Vec2D:
        """
        `col` and `row` are the column and row numbers associated with the tile,
        starting with 0.
        `position_in_tile` describes, from `<0, 0>` to `<1, 1>`, where within
        the tile the ouputted coordinate will be. The default is `<0.5, 0.5>`.
        """
        if col < 0 or col >= self.num_tile_cols:
            raise ValueError("Column number out of range")
        elif row < 0 or row >= self.num_tile_rows:
            raise ValueError("Row number out of range")
        
        grid_across = self.edge_coords[1] - self.edge_coords[0]
        tile_size = grid_across.multiply_components(1.0/self.num_tile_cols, 1.0/self.num_tile_rows)
        
        return self.edge_coords[0] + tile_size.multiply_components(position_in_tile + Vec2D(col, row))
    
    def calibrate_grid(self):
        # Find flag for reference
        flag_x, flag_y = self.__locate_flag()
        self.flag_icon_position = Vec2D(flag_x, flag_y)
        
        # Stillness before grid reference screenshot
        self.debug_print(f"Moving mouse to flag at ({flag_x}, {flag_y})")
        pyautogui.moveTo(flag_x, flag_y)
        pyautogui.sleep(0.2)
        screen_view = pyautogui.screenshot()
        start_x, start_y = [flag_x - 50, flag_y]
        
        grid_top_y = self.__get_grid_top_y(screen_view, start_x, start_y)
        grid_left_x, grid_right_x = self.__get_grid_left_right_xs(screen_view, start_x, grid_top_y)
        grid_bottom_y = self.__get_grid_bottom_y(screen_view, grid_left_x, grid_top_y)
        
        self.edge_coords = [
            Vec2D(grid_left_x, grid_top_y),
            Vec2D(grid_right_x, grid_bottom_y),
        ]
        
        if self.debug:
            print(f"Grid coordinates found: top left {self.edge_coords[0]} to bottom right {self.edge_coords[1]}.")
        
        self.num_tile_cols = self.__get_num_tiles(screen_view, 0, grid_left_x, grid_right_x, grid_top_y)
        self.num_tile_rows = self.__get_num_tiles(screen_view, 1, grid_top_y, grid_bottom_y, grid_left_x)
        
        print("Game grid successfully calibrated.")
    
    def __get_grid_top_y(self, screen_view: Image.Image, start_x: int, start_y: int) -> int:
        game_top_bar_color = Color.from_pixel(screen_view, start_x, start_y)
        for y in range(start_y, pyautogui.size().height):
            if Color.from_pixel(screen_view, start_x, y) != game_top_bar_color:
                if self.debug:
                    pyautogui.moveTo(start_x, start_y)
                    pyautogui.moveTo(start_x, y, duration=0.5)
                return y
        raise MinesweeperException("Could not find top edge of minesweeper grid")
    
    def __get_grid_left_right_xs(self, screen_view: Image.Image, start_x: int, grid_top_y: int) -> tuple[int, int, int]:
        color_brightness = lambda c: max(c.r, c.g, c.b)
        left_x = None
        right_x = None
        
        # Dark background to left and right of grid
        
        # Find left edge
        y = grid_top_y
        for x in range(start_x, 0, -1):
            color = Color.from_pixel(screen_view, x, y)
            if color_brightness(color) < 60:
                left_x = x + 1
                break
        if left_x is None:
            raise MinesweeperException("Could not find left edge of minesweeper grid")
        
        if self.debug:
            pyautogui.moveTo(start_x, y)
            pyautogui.moveTo(left_x, y, duration=0.5)
        
        for x in range(left_x, pyautogui.size().width, 1):
            color = Color.from_pixel(screen_view, x, y)
            if color_brightness(color) < 50:
                right_x = x - 1
                break
        if right_x is None:
            raise MinesweeperException("Could not find right edge of minesweeper grid")
        
        if self.debug:
            pyautogui.moveTo(right_x, y, duration=0.5)
        
        return (left_x, right_x)
    
    def __get_grid_bottom_y(self, screen_view: Image.Image, grid_left_x: int, grid_top_y: int) -> int:
        color_brightness = lambda c: max(c.r, c.g, c.b)
        
        # Dark background underneath grid
        x = grid_left_x
        bottom_y = None
        for y in range(grid_top_y, pyautogui.size().height):
            color = Color.from_pixel(screen_view, x, y)
            if color_brightness(color) < 50:
                if self.debug:
                    pyautogui.moveTo(x, grid_top_y)
                    pyautogui.moveTo(x, y - 1, duration=0.5)
                bottom_y = y - 1
                break
        if bottom_y is None:
            raise MinesweeperException("Could not find bottom edge of grid")
        
        return bottom_y
    
    def __get_calibrated_grid_dimension(self, screen_view: Image.Image, start_x: int, start_y: int, dimension: int) -> tuple[int, int, int]:
        """
        Tile color at `[start_x, start_y]` assumed to be `self.tileColor1`
        Dimension is 0 for x, 1 for y
        Returns [minCoord, maxCoord, numBoxes]
        """
        
        start_coords = [start_x, start_y]
        
        screen_size = pyautogui.size()
        screen_dim_size = [screen_size.width, screen_size.height][dimension]
        
        # Find negative edge of coordinate grid
        min_coord_search_range = range(start_coords[dimension], 0, -1)
        min_coord, num_boxes_negative_side = self.__find_coord_grid_edge(screen_view, start_coords, dimension, min_coord_search_range)
        
        # Find positive edge of coordinate grid
        max_coord_search_range = range(start_coords[dimension], screen_dim_size, 1)
        max_coord, num_boxes_positive_side = self.__find_coord_grid_edge(screen_view, start_coords, dimension, max_coord_search_range)
        
        # Put it all together
        num_boxes = num_boxes_negative_side + 1 + num_boxes_positive_side
        return [min_coord, max_coord, num_boxes]
    
    def __find_coord_grid_edge(self, screen_view: Image.Image, start_coords: tuple[int, int], dimension: int, search_coord_range: range):
        """
        See inputs to `_get_calibrated_grid_dimension()`
        Returns `[coord_edge, num_boxes_found]`. `num_boxes_found` does NOT include first box
        
        Basically loops through the given search_coord_range counting tiles/boxes and finding the extreme edge of the coordinate grid
        """
        
        # Find negative edge of coordinate grid
        num_boxes_found = 0
        tileColorIdx = 0 # Start with self.tileColor1 (idx:0) at (x, y) = [start_x, start_y]
        
        coord_edge = None
        
        for v in search_coord_range:
            coords = copy.copy(start_coords)
            coords[dimension] = v
            x, y = coords
            
            # Get the color of the current pixel and the colors of the "current" (last) tile and the "next" (expected next) tile
            color = Color.from_pixel(screen_view, x, y)
            currentTileColor = self.tileColor1 if tileColorIdx == 0 else self.tileColor2
            nextTileColor = self.tileColor2 if tileColorIdx == 0 else self.tileColor1
            
            # Check which tile type this color matches
            if color != currentTileColor:
                if color == nextTileColor:
                    # Current color matches next tile: change the tile index and increase number of boxes
                    num_boxes_found += 1
                    tileColorIdx = (tileColorIdx + 1) % 2
                else:
                    # Current color matches neither tile--edge of grid
                    coord_edge = v
                    break
            else:
                # Nothing happens! We found the same tile as last time. boooring
                pass
        return [coord_edge, num_boxes_found]
    
    def __get_num_tiles(self, screen_view: Image.Image, component_idx: int, coord_start: int, coord_end: int, constant_coord: int):
        xy = [0, 0]
        xy[(component_idx + 1) % 2] = constant_coord
        
        num_tiles = 1
        tile_color: Color|None = None
        
        coord = coord_start
        while coord <= pyautogui.size()[component_idx] and coord < coord_end:
            xy[component_idx] = coord
            pixel_color = Color.from_pixel(screen_view, xy[0], xy[1])
            
            if tile_color == None:
                # If the tile color is unset, set it and continue to the next pixel
                tile_color = pixel_color
                coord += 1
            elif pixel_color == tile_color:
                # If the tile color is set and this pixel is the same, continue
                # to the next pixel
                coord += 1
            else:
                # We've reached (roughly) the end of the tile (generally, besides borders)
                # Increase the number of tiles and start at roughly halfway through the next tile
                num_tiles += 1
                tile_color = None
                tile_width_approx = float(coord - coord_start) / float(num_tiles)
                coord = int(tile_width_approx*0.5) + coord + 1
        
        if self.debug:
            comp = "x" if component_idx == 0 else "y"
            print(f"Num tiles along {comp}-axis: {num_tiles}")
            xy[component_idx] = coord_start
            pyautogui.moveTo(xy)
            xy[component_idx] = coord_end
            pyautogui.moveTo(xy, duration=0.5)
        
        return num_tiles
    
    def __update_tile_colors(self, screen_view: Image.Image, start_x: int, start_y: int):
        # First, establish tileColor1
        tileColor1 = Color.from_pixel(screen_view, start_x, start_y)
        tileColor2 = None
        self.debug_print(f"Tile color #1: {tileColor1}")
        if self.debug:
            pyautogui.moveTo(start_x, start_y)
        
        # Next, find tileColor2, moving to right until a new color is found
        for x in range(start_x, pyautogui.size().width):
            color = Color.from_pixel(screen_view, x, start_y)
            if color != tileColor1:
                tileColor2 = color
                self.debug_print(f"Tile color #2: {tileColor2}")
                if self.debug:
                    pyautogui.moveTo(x, start_y)
                break
        if tileColor2 is None:
            raise MinesweeperException("Could not find tile color #2.")
        
        self.tileColor1 = tileColor1
        self.tileColor2 = tileColor2

    def __locate_flag(self) -> tuple[int, int]:
        self.debug_print("Locating flag...")
        result = None
        try:
            result = pyautogui.locateOnScreen(Resources.FLAG_ICON.value)
            x, y = [result.left + result.width / 2, result.top + result.height / 2]
            return [int(x), int(y)]
        except pyautogui.ImageNotFoundException:
            raise MinesweeperException("Flag not found!")
    
    def debug_print(self, content):
        if self.debug:
            print(content)

TILE_STATE = Literal["GREEN"] | Literal["MINE"] | int | Literal["UNCOVER"]
"""
Tile state of `"GREEN"` means the tile is not yet discovered. Tile state
of `"MINE"` means the tile is a mine. Tile state of int (0 through 8)
means the tile is not a bomb and has a number associated with it.

Tile state of `"UNCOVER"` is not a true tile state in minesweeper,
but is used for tiles which are known not to be mines or at least ought to be
clicked / uncovered (best bets if no better deduction can be made).
"""

class MinesweeperGameState:
    __VALID_GAME_STATES = [
        ("Easy", 10, 8, 10),
        ("Medium", 18, 14, 40),
        ("Hard", 24, 20, 99),
    ]
    
    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        
        states = list(filter(lambda state: state[1] == self.cols and state[2] == self.rows, self.__VALID_GAME_STATES))
        if len(states) == 0:
            raise MinesweeperException(f"No valid game type has columns and rows {self.cols} x {self.rows}")
        self.game_mode_name: str = states[0][0]
        self.num_mines: int = states[0][3]
        
        self.__tile_states: list[list[TILE_STATE]] = [[
            "GREEN" for row_num in range(self.rows)
        ] for col_num in range(self.cols) ]
        """
        Tile states is indexed by col then row, e.g. `__tile_states[col][row]`
        """
        
        self.__any_moves_made = False
    
    def get_grid_string_representation(self) -> str:
        grid_str = GridString()
        
        x_mul = 2
        y_mul = 1
        
        # Draw col and row markers
        for col in range(self.cols):
            grid_str.put_str(col, col * x_mul + x_mul-1, -2, dir="down", justify="end")
        for row in range(self.rows):
            grid_str.put_str(row, -3, row * y_mul, dir="right", justify="end")
        grid_str.put_str("|"*(self.rows*y_mul + 1), -1, -1, dir="down")
        grid_str.put_str("_"*(self.cols*x_mul + 1), -1, -1)
        
        for col, row in self.get_all_positions():
            tile_state = self.get_tile(col, row)
            x_pad = " "*(x_mul-1)
            
            repr = x_pad
            if isinstance(tile_state, int):
                repr += str(tile_state)
            elif tile_state == "MINE":
                repr += "#"
            elif tile_state == "UNCOVER":
                repr += "U"
            else:
                repr += "."
            
            grid_str.put_str(repr, col*x_mul, row*y_mul)
        
        return grid_str.get_full_string()
    
    def get_tile(self, col: int, row: int) -> TILE_STATE:
        """
        Zero-indexed col and row. Get the state of the tile.
        """
        if col < 0 or col >= self.cols:
            raise ValueError("Column out of range")
        elif row < 0 or row >= self.rows:
            raise ValueError("Row out of range")
        return self.__tile_states[col][row]
    
    def set_tile(self, col: int, row: int, state: TILE_STATE):
        """
        Zero-indexed col and row. Get the state of the tile.
        """
        if col < 0 or col >= self.cols:
            raise ValueError("Column out of range")
        elif row < 0 or row >= self.rows:
            raise ValueError("Row out of range")
        
        if state != "GREEN" and state != "MINE":
            self.__any_moves_made = True
        
        self.__tile_states[col][row] = state
    
    def has_move_been_made(self) -> bool:
        return self.__any_moves_made
    
    def get_all_positions(self):
        for col in range(self.cols):
            for row in range(self.rows):
                yield (col, row)
    
    def get_neighborhood(self, col: int, row: int):
        min_col, max_col = max(0, col - 1), min(self.cols, col + 2)
        min_row, max_row = max(0, row - 1), min(self.rows, row + 2)
        
        for nc in range(min_col, max_col):
            for nr in range(min_row, max_row):
                if nc != col or nr != row:
                    yield (nc, nr)
    
    def in_neighborhood(self, host_col: int, host_row: int, target_col: int, target_row: int) -> bool:
        col_diff = abs(host_col - target_col)
        row_diff = abs(host_row - target_row)
        return col_diff <= 1 and row_diff <= 1
    
    def __try_uncover_middle_tile(self) -> bool:
        """
        Solve strategy for beginning of game--just uncover a tile
        in the middle of the game board
        """
        # If no moves have been made, then uncover a tile in the middle of the board
        if not self.__any_moves_made:
            self.set_tile(int(self.cols / 2), int(self.rows / 2), "UNCOVER")
            return True
        else:
            return False
    
    def __get_tile_neighbor_stats(self, col, row) -> tuple[int, int, int]:
        """
        Return (numbers, mines, greens) numbers of neighboring tile types
        """
        num_number_neighbors = 0
        num_mine_neighbors = 0
        num_green_neighbors = 0
        
        # Check number of green neighboring tiles
        for nc, nr in self.get_neighborhood(col, row):
            state = self.get_tile(nc, nr)
            if state == "GREEN":
                num_green_neighbors += 1
            elif state == "MINE":
                num_mine_neighbors += 1
            else:
                num_number_neighbors += 1
        
        return (num_number_neighbors, num_mine_neighbors, num_green_neighbors)
    
    def __try_mine_saturated_tiles(self) -> bool:
        """
        Find number tiles with the exact correct number of mines around them.
        """
        move_made = False
        for col, row in self.get_all_positions():
            host_tile_state = self.get_tile(col, row)
            if not isinstance(host_tile_state, int):
                continue
            
            nums, mines, greens = self.__get_tile_neighbor_stats(col, row)
            if mines == host_tile_state:
                for nc, nr in self.get_neighborhood(col, row):
                    if self.get_tile(nc, nr) == "GREEN":
                        self.set_tile(nc, nr, "UNCOVER")
                        move_made = True
        return move_made
        
    def __try_green_saturated_tiles(self) -> bool:
        """
        Find number tiles which need all remaining green tiles to be mines in
        order to fulfill their numbers.
        """
        move_made = False
        for col, row in self.get_all_positions():
            host_tile_state = self.get_tile(col, row)
            if not isinstance(host_tile_state, int):
                continue
            
            nums, mines, greens = self.__get_tile_neighbor_stats(col, row)
            if greens + mines == host_tile_state:
                for nc, nr in self.get_neighborhood(col, row):
                    if self.get_tile(nc, nr) == "GREEN":
                        self.set_tile(nc, nr, "MINE")
                        move_made = True
        return move_made
    
    # TODO: Optimization for searching
    # 1) Create an ordered "tile frontier" of where to actually search
    #    for deductions
    # 2) Also create a list of animations and the boxes of tiles they affect,
    #    and order the frontier based on when areas around tiles will be available
    #    for screenshot analysis
    # 3) Write the frontier code such that we can see which tiles need to be
    #    re-checked at any given point, based on recent uncoverings
    
    def __try_possibilities_reduction(self, debug:tuple[int,int,int,int]|None=None) -> bool:
        move_made = False
        
        for icol, irow in self.get_all_positions():
            if debug is not None:
                icol, irow = debug[2:4]
            
            intermediate_tile_state = self.get_tile(icol, irow)
            if not isinstance(intermediate_tile_state, int):
                continue
            
            _, imines, _ = self.__get_tile_neighbor_stats(icol, irow)
            igreens_locations = []
            for nc, nr in self.get_neighborhood(icol, irow):
                if self.get_tile(nc, nr) == "GREEN":
                    igreens_locations.append((nc, nr))
            
            if len(igreens_locations) == 0:
                continue
            
            for hcol in range(icol - 2, icol + 3):
                if hcol < 0 or hcol >= self.cols:
                    continue
                for hrow in range(irow - 2, irow + 3):
                    if hrow < 0 or hrow >= self.rows:
                        continue
                    elif icol == hcol and irow == hrow:
                        continue
                    
                    if debug is not None:
                        hcol, hrow = debug[0:2]
                    
                    if self.__try_possibilities_reduction_at(hcol, hrow, intermediate_tile_state, imines, igreens_locations, debug=(debug is not None)):
                        move_made = True
                    
                    if debug is not None:
                        return
        
        return move_made
    
    # TODO: new strategy - binary tile state reducer
    # 1) An "important" green tile is found
    # 2) A new hypothetical child board is created with that green tile as a mine
    #    and is reduced
    # 3) A hypothetical child board is created with that tile as uncovered and is reduced
    # 4) Any tiles in agreement between the two boards are marked in the parent game state
    # Caveats:
    # - Requires recursion checks, should run reductions of the binary children
    #   which themselves do not run binary tile state reduction
    # - Can replace some (or maybe all? figure out) "possibilities reductions"
    
    # TODO:
    # Strategy:
    # - Something based on total number of mines
    # - Eventual strategy involving taking a chance
    # Polishings:
    # - Command line options, particularly on flag placement, eventually
    #   on whether imperfect or risky moves are allowed, eventually on
    #   interaction movement (mouse control, etc.)
    # - EVENTUALLY recognizing game completion or failure
    # - Printing finishing info: whether the game was completed,
    #   whether the game failed, and timing -- how long the solver took
    # - Better tile recognition around board with new screenshot strategy
    # - Another control layer between interaction and hardware allows for
    #   controlling mouse movement and clicks, maybe
    
    def __try_possibilities_reduction_at(self, hcol: int, hrow: int, itile_state: int, imines: int, igreens_locations: list[tuple[int, int]], debug:bool=False):
        host_tile_state = self.get_tile(hcol, hrow)
        move_made = False
        
        if debug:
            print(f"Host tile state: {host_tile_state}, inter tile state: {itile_state}")
            print(f"Inter's number of neighboring mines: {imines}")
            print(f"Inter's green tiles: {igreens_locations}")
        
        if not isinstance(host_tile_state, int):
            return False
        
        # Check if ALL intermediate's green tiles are within
        # this host's neighborhood
        if False in [ self.in_neighborhood(hcol, hrow, nc, nr) for nc, nr in igreens_locations ]:
            return False
        
        # If all the intermediate's green tiles ARE within this host's
        # neighborhood, then any remaining mines count towards this tile
        hnums, hmines, hgreens = self.__get_tile_neighbor_stats(hcol, hrow)
        inter_remaining_mines = itile_state - imines
        
        if debug:
            print(f"Host's number of neighboring mines: {hmines}")
        
        # TOTAL number of mines left to find for the host
        host_total_remaining_mines = host_tile_state - hmines
        # Number of mines left to find for the host OUTSIDE the intermediate's neighborhood
        host_rmines_after_intermediate = host_total_remaining_mines - inter_remaining_mines
        
        # Get list of host's green tiles which aren't in the intermediate's neighborhood
        host_greens_outside_intermediate = []
        for nc, nr in self.get_neighborhood(hcol, hrow):
            if self.get_tile(nc, nr) == "GREEN" and not ((nc, nr) in igreens_locations):
                host_greens_outside_intermediate.append((nc, nr))
        if len(host_greens_outside_intermediate) == 0:
            return False
        
        # TODO: Could CHAIN "optionals" (possibilities for mine locations) here, even if
        # we can't find any definitive moves on this host, but implement this later
        if host_rmines_after_intermediate == len(host_greens_outside_intermediate):
            # All mines remaining in host's greens outside the intermediate
            for tc, tr in host_greens_outside_intermediate:
                self.set_tile(tc, tr, "MINE")
                move_made = True
                if debug:
                    print(f"Found mine at {tc}x{tr}")
                
        elif host_rmines_after_intermediate == 0:
            # No mines remaining in host's greens outside the intermediate
            for tc, tr in host_greens_outside_intermediate:
                self.set_tile(tc, tr, "UNCOVER")
                move_made = True
                if debug:
                    print(f"Uncovered tile at {tc}x{tr}")
        
        return move_made
    
    def __try_binary_option_reduction(self, recursion_limit: int) -> bool:
        # Step 1 - Find and order good target tiles for binary reduction
        target_tiles: list[tuple[int, int, int]] = [] # (score, col, row) with maximal score for best targets
        for col, row in self.get_all_positions():
            state = self.get_tile(col, row)
            if state != "GREEN":
                continue
            
            nums, mines, greens = self.__get_tile_neighbor_stats(col, row)
            # We just allow the score of the tile to be the number of neighboring
            # number tiles--this should target tiles with lots of impact on state
            score = nums
            
            # We have to assign a practical cutoff, and having at least 3 neighbors
            # seems alright
            if score >= 2:
                target_tiles.append((score, col, row))
        
        # Step 2 - walk through potential targets and see if we get anything.
        # Keep trying until we get some moves made, but if we do make a move,
        # we should return to less computationally intense strategies
        for score, col, row in target_tiles:
            move_made = self.__try_binary_option_reduction_at(recursion_limit, col, row)
            if move_made:
                return True
        return False
    
    def __try_binary_option_reduction_at(self, recursion_limit: int, col: int, row: int):
        # Create two child states
        child_state_uncover = copy.deepcopy(self)
        child_state_mine = copy.deepcopy(self)
        
        # Set the tile
        child_state_uncover.set_tile(col, row, "UNCOVER")
        child_state_mine.set_tile(col, row, "MINE")
        
        # Attempt reduction-ception
        child_state_uncover.reduce_state(recursion_limit-1)
        child_state_mine.reduce_state(recursion_limit-1)
        
        # Loop through tiles and find commonalities
        move_made = False
        for c, r in self.get_all_positions():
            known_state = self.get_tile(c, r)
            state_1 = child_state_uncover.get_tile(c, r)
            state_2 = child_state_mine.get_tile(c, r)
            if known_state == state_1 and state_1 == state_2:
                # Nothing to see here. Boring!
                continue
            elif known_state != state_1 and state_1 == state_2:
                # New state -- both binary options lead to shared state
                self.set_tile(c, r, state_1)
                move_made = True
        return move_made
    
    def reduce_state_single_step(self, slow_options_ok: bool, binary_option_recursion_limit: int) -> bool:
        """
        Perform one step of game board state reduction.
        Returns `true` if more reduction should be completed,
        `false` otherwise.
        """
        
        move_made = False
        
        move_made = self.__try_uncover_middle_tile() or move_made
        move_made = self.__try_green_saturated_tiles() or move_made
        move_made = self.__try_mine_saturated_tiles() or move_made
        if not move_made:
            move_made = self.__try_possibilities_reduction()
        if slow_options_ok and not move_made and binary_option_recursion_limit > 0:
            move_made = self.__try_binary_option_reduction(binary_option_recursion_limit)
        
        return move_made
    
    def reduce_state(self, binary_option_recursion_limit:int=2) -> bool:
        """
        Reduce the state of the game board.
        """
        slow_options_ok = False
        any_move_made = False
        max_iters = 100
        for _ in range(max_iters):
            continue_reduction = self.reduce_state_single_step(slow_options_ok, binary_option_recursion_limit)
            if continue_reduction:
                slow_options_ok = False
                any_move_made = True
            else:
                if any_move_made or slow_options_ok:
                    # Just stop, if either we've already tried
                    # slow options or if we've already made a move
                    # (slow options not necessary)
                    break
                else:
                    slow_options_ok = True
        return any_move_made

class GameGridInteractionLayer:
    __ANIM_TIME_REMOVE_FLAG = 4.6
    # TODO: Carefully determined time, could make algorithm for obscured vision
    # by searching through all pixels in a box and finding pixels with lowest diffs
    # to known colors
    __ANIM_TIME_UNCOVER = 1.05
    __ANIM_TIME_ADD_FLAG = 0.1
    
    def __init__(self, grid: GameGridCalibration):
        self.grid = grid
        self.cols = grid.num_tile_cols
        self.rows = grid.num_tile_rows
        self.__true_game_state = MinesweeperGameState(self.cols, self.rows)
        for col, row in self.__true_game_state.get_all_positions():
            self.__true_game_state.set_tile(col, row, "UNCOVER")
        self.__last_game_screenshot: Image.Image|None = None
        self.__time_to_last_animation_finish = datetime.datetime.now()
    
    def get_true_game_state_representation(self) -> str:
        # Just run through all tiles and make sure we have their values
        for col, row in self.__true_game_state.get_all_positions():
            self.get_tile(col, row)
        return self.__true_game_state.get_grid_string_representation()
    
    def __game_action_animation_time(self, animation_time_seconds: float):
        """
        To be called internally when an action is taken where an animation could interfere
        with the correct judgement of the game state from a screenshot. The duration of the
        animation should be passed into the function so we can wait the correct amount of
        time before taking a screenshot.
        """
        ttf = datetime.datetime.now() + datetime.timedelta(seconds=animation_time_seconds)
        if ttf > self.__time_to_last_animation_finish:
            self.__time_to_last_animation_finish = ttf
    
    def __get_new_game_screenshot(self) -> Image.Image:
        pyautogui.moveTo(self.grid.flag_icon_position.x, self.grid.flag_icon_position.y)
        wait_time = (self.__time_to_last_animation_finish - datetime.datetime.now()).total_seconds()
        if wait_time > 0:
            pyautogui.sleep(wait_time)
        return pyautogui.screenshot()
    
    def __tile_lookup(self, col: int, row: int) -> TILE_STATE:
        """
        Look up the true tile state from the last screenshot available
        """
        
        if self.__last_game_screenshot == None:
            self.__last_game_screenshot = self.__get_new_game_screenshot()
        
        # Get the integer [x0, y0] and [x1, y1] coords describing where in the tile to search
        p_from_corners = 0.15
        tile_start_vec = self.grid.get_position(col, row, Vec2D(p_from_corners, p_from_corners))
        tile_end_vec = self.grid.get_position(col, row, Vec2D(1-p_from_corners, 1-p_from_corners))
        
        x0, y0 = tile_start_vec.components()
        x0, y0 = [int(round(x0)), int(round(y0))]
        x1, y1 = tile_end_vec.components()
        x1, y1 = [int(round(x1)), int(round(y1))]
        
        # The base color of the tile
        base_color = None
        current_content_x_start = None
        
        # We take several ranges and find the largest. This can be necessary
        # for content (like the 5) which has short edges near the middle, which
        # can lead to misreads
        content_x_ranges = []
        
        y = int(round((y1 - y0) / 2 + y0))
        for x in range(x0, x1):
            
            # Extract the pixel color from the screenshot
            pixel_color = Color.from_pixel(self.__last_game_screenshot, x, y)
            
            # Get the base color of the tile if it hasn't been found
            if base_color == None:
                base_color = pixel_color
            
            # Record the start and end of the cross-section of the content
            # to later find the color of its center
            if pixel_color != base_color and current_content_x_start is None:
                # Non-base color pixel and content_x_start not yet found -- set it
                current_content_x_start = x
            if pixel_color == base_color and current_content_x_start is not None:
                # Base color pixel and content_x_start found -- end has been found
                content_x_ranges.append((current_content_x_start, x))
                current_content_x_start = None
        
        if current_content_x_start is not None:
            content_x_ranges.append((current_content_x_start, x1))
        
        # The color of the foreground, if a foreground exists
        content_color = None
        
        # Get the longest range from the content x ranges
        content_x_start = 0
        content_x_end = 0
        for x_start, x_end in content_x_ranges:
            if x_end - x_start > content_x_end - content_x_start:
                content_x_start, content_x_end = x_start, x_end
        
        # Check if there was no content
        if content_x_start == content_x_end:
            content_x_start, content_x_end = None, None
        
        # If a foreground exists
        if content_x_start is not None:
            
            # Get the center of the foreground
            content_x_center = (content_x_end - content_x_start) / 2 + content_x_start
            content_color = Color.from_pixel(self.__last_game_screenshot, content_x_center, y)
        
        return self.__tile_colors_lookup(base_color, content_color)
    
    def __tile_colors_lookup(self, base_color: Color, content_color: Color|None) -> TILE_STATE:
        if base_color.compare(Resources.COLOR_GREEN_TILE_LIGHT.value, 17) or base_color.compare(Resources.COLOR_GREEN_TILE_DARK.value, 17):
            # Green tile background
            if content_color == None:
                return "GREEN"
            else:
                # We can just assume the tile is a mine
                return "MINE"
        else:
            # Non-green tile
            if content_color == None:
                # Empty tile - 0 mines around
                return 0
            else:
                # TODO: How to deal with antialiasing
                # Match against colors for numbers
                colors_list = [
                    Resources.COLOR_MINES_1.value,
                    Resources.COLOR_MINES_2.value,
                    Resources.COLOR_MINES_3.value,
                    Resources.COLOR_MINES_4.value,
                    Resources.COLOR_MINES_5.value,
                    Resources.COLOR_MINES_6.value,
                    Resources.COLOR_MINES_7.value,
                    Resources.COLOR_MINES_8.value,
                ]
                
                idx = content_color.get_best_match(colors_list)
                mines_num = idx + 1
                return mines_num
    
    def __tile_uncover_update(self):
        """
        A tile was uncovered, so all MINE tiles and GREEN tiles here
        become UNCOVER, which in this context signifies that we simply
        haven't looked them up yet.
        """
        # The last game state screenshot does not apply anymore
        self.__last_game_screenshot = None
        
        # Set all MINE or GREEN tiles to UNCOVER, signifying that they're not yet known
        for col, row in self.__true_game_state.get_all_positions():
            current_tile_state: TILE_STATE = self.__true_game_state.get_tile(col, row)
            if current_tile_state == "MINE" or current_tile_state == "GREEN":
                self.__true_game_state.set_tile(col, row, "UNCOVER")
    
    def get_tile(self, col: int, row: int) -> TILE_STATE:
        """
        Get the true state of a tile.
        A return type of `int` indicates the number of adjascent mines.
        A return value of `"MINE"` indicates the tile is flagged, and `"GREEN"` indicates the tile is
        green without a flag.
        The value `"UNCOVER"` will never be returned.
        """
        
        # Look up the current tile state
        lookup_tile_state = self.__true_game_state.get_tile(col, row)
        
        if lookup_tile_state == "UNCOVER":
            # UNCOVER here signifies that the true tile state isn't known
            # We get the true tile state
            tile_state = self.__tile_lookup(col, row)
            # And we update the __true_game_state to match
            self.__true_game_state.set_tile(col, row, tile_state)
            return tile_state
        else:
            # Tile state is known (not UNCOVER)
            return lookup_tile_state
    
    def __click_tile(self, col: int, row: int, button):
        location = self.grid.get_position(col, row)
        pyautogui.click(location.x, location.y, button=button)
    
    def uncover_tiles(self, tiles: list[tuple[int, int]]):
        """
        Uncover (left click) a given tile. Fails if the tile is not GREEN or MINE
        (i.e. if it's not an actionable tile).
        """
        
        tiles = [ (col, row, self.get_tile(col, row)) for col, row in tiles ]
        
        for col, row, tile_state in tiles:
            # Right click the tile if it has a flag on it to remove the flag first
            if tile_state == "MINE":
                self.__click_tile(col, row, "RIGHT")
                self.__game_action_animation_time(self.__ANIM_TIME_REMOVE_FLAG)
            
            # Click the tile to uncover
            self.__click_tile(col, row, "LEFT")
            self.__game_action_animation_time(self.__ANIM_TIME_UNCOVER)
            
            # Update the game status so we know that a tile was uncovered
            self.__tile_uncover_update()
    
    def set_flagged_status(self, col: int, row: int, flagged_status: bool):
        """
        Set whether or not we want a mine flag to be placed at a specific tile.
        Fails if the tile is not GREEN or MINE (i.e. if it's not an actionable tile)
        """
        
        # Check if we can set whether a flag is placed here
        tile_state = self.get_tile(col, row)
        if tile_state != "GREEN" and tile_state != "MINE":
            raise MinesweeperException(f"Cannot set flag state on non-green or non-flag tile: {col} x {row}")
        
        # Only perform an action if we want to actually change the game
        # state from flag to no flag or no flag to flag
        if flagged_status != (tile_state == "MINE"):
            # Right click the tile
            self.__click_tile(col, row, "RIGHT")
            self.__game_action_animation_time(self.__ANIM_TIME_ADD_FLAG if flagged_status else self.__ANIM_TIME_REMOVE_FLAG)
            # Update the game state
            self.__true_game_state.set_tile(col, row, "MINE" if flagged_status else "GREEN")

class GameSolver:
    def __init__(self, interact: GameGridInteractionLayer):
        self.interact = interact
        self.game_state = MinesweeperGameState(self.interact.cols, self.interact.rows)
        
        # Read tile states
        self.__update_true_game_state()
        game_desc = f"'{self.game_state.game_mode_name}' {self.game_state.cols} x {self.game_state.rows} game with {self.game_state.num_mines} total mines"
        if not self.game_state.has_move_been_made():
            print(f"Beginning solve of {game_desc}.")
        else:
            print(f"Attempting to complete partial solve of {game_desc}.")
            print("\nInitial grid state:")
            print(self.interact.get_true_game_state_representation())
    
    def __update_true_game_state(self):
        for col, row in self.game_state.get_all_positions():
            true_state = self.interact.get_tile(col, row)
            if isinstance(true_state, int):
                self.game_state.set_tile(col, row, true_state)
    
    def __interact_update_tile_states(self):
        uncover_tiles = []
        tiles_flag_states = []
        for col, row in self.game_state.get_all_positions():
            state = self.game_state.get_tile(col, row)
            if state == "UNCOVER":
                uncover_tiles.append((col, row))
            elif state == "MINE":
                tiles_flag_states.append((col, row, True))
            elif state == "GREEN":
                tiles_flag_states.append((col, row, False))
        
        for col, row, flag in tiles_flag_states:
            self.interact.set_flagged_status(col, row, flag)
        self.interact.uncover_tiles(uncover_tiles)
    
    def solve(self):
        continue_cycle = True
        while continue_cycle:
            continue_cycle = self.__solve_cycle()
    
    def __solve_cycle(self) -> bool:
        
        # Update and reduce the game state
        self.__update_true_game_state()
        any_move_made = self.game_state.reduce_state()
        
        # Look up mines to flag (or unflag if green) and boxes to uncover
        self.__interact_update_tile_states()
        return any_move_made

def sweep_mines(debug: bool) -> bool:
    try:
        grid = GameGridCalibration(debug)
        interact = GameGridInteractionLayer(grid)
        solver = GameSolver(interact)
        
        solver.solve()
        
        # # TODO: Add time to complete from first click
        
        return True
    except pyautogui.FailSafeException:
        print("Error! Moving the mouse to the top-left corner of")
        print("of the screen disables GUI control.")
        return False

def __wait_for_spacebar():
    listener = None
    
    def get_key_press(key):
        if key == pynput.keyboard.Key.space:
            listener.stop()
    
    listener = pynput.keyboard.Listener(on_press=get_key_press)
    listener.start()
    listener.join()

def __wait_sweep_mines(debug: bool):
    print("When you're ready to start the minesweeper solver,")
    print("press enter into this terminal. You'll have three")
    print("seconds to move or maximize the Google minesweeper")
    print("window after this starts.")
    # __wait_for_spacebar()
    input()
    print("3 seconds before start...\n")
    pyautogui.sleep(3)
    
    try:
        return sweep_mines(debug)
    except MinesweeperException as e:
        print(f"Error! {e}")
        return False

def main(args: list[str]) -> bool:
    debug = False
    
    if len(args) == 0:
        pass
    else:
        if args[0] == "debug":
            debug = True
        else:
            print(f"Invalid argument: {args[0]}")
            return False
    
    return __wait_sweep_mines(debug)

if __name__ == "__main__":
    try:
        exit(0 if main(sys.argv[1:]) else 1)
    except KeyboardInterrupt:
        exit(1)
