import copy
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
        # TODO: PIL image getpixel color doesn't agree
        # with gpick color picker from image color OR
        # imagemagick color analyzer from saved PNG
        return Color.from_rgb(image.getpixel([x, y]))
    
    def __repr__(self):
        return f"Color#{hex(self.hex_code)[2:].upper()}"
    
    def __get_quantitative_diff(self, other) -> int:
        c1 = [self.r, self.g, self.b]
        c2 = [other.r, other.g, other.b]
        diffs = [ (c1[i]-c2[i])**2 for i in range(3) ]
        return diffs[0] + diffs[1] + diffs[2]
    
    def compare(self, other, tol:int=17) -> bool:
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
            return self.compare(other)
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

class GameGridCalibration:
    tileColor1: Color
    tileColor2: Color
    
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
        time.sleep(0.2)
        screen_view = pyautogui.screenshot()
        start_x, start_y = [flag_x, flag_y + 50]
        
        # Update self.tileColor1 and tileColor2
        self.__update_tile_colors(screen_view, start_x, start_y)
        
        # Calibrate X and Y grid dimensions
        x1, x2, self.num_tile_cols = self.__get_calibrated_grid_dimension(screen_view, start_x, start_y, 0)
        y1, y2, self.num_tile_rows = self.__get_calibrated_grid_dimension(screen_view, start_x, start_y, 1)
        self.edge_coords = [Vec2D(x1, y1), Vec2D(x2, y2)]
        
        if self.debug:
            x1, y1 = self.edge_coords[0].components()
            x2, y2 = self.edge_coords[1].components()
            print(f"Grid coordinates found: top left {self.edge_coords[0]} to bottom right {self.edge_coords[1]}.")
            print(f"Tiles are {self.num_tile_cols} x {self.num_tile_rows}.")
            
            pyautogui.moveTo(x1, y1)
            d = 0.75
            pyautogui.moveTo(x2, y1, duration=d)
            pyautogui.moveTo(x2, y2, duration=d)
            pyautogui.moveTo(x1, y2, duration=d)
            pyautogui.moveTo(x1, y1, duration=d)
        print("Game grid successfully calibrated.")
    
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
    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        
        self.__tile_states: list[list[TILE_STATE]] = [[
            "GREEN" for row_num in range(self.rows)
        ] for col_num in range(self.cols) ]
        """
        Tile states is indexed by col then row, e.g. `__tile_states[col][row]`
        """
        
        self.__any_moves_made = False
    
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
        
        if state != "GREEN":
            self.__any_moves_made = True
        
        self.__tile_states[col][row] = state
    
    def reduce_state(self):
        """
        Reduce the state of the game board.
        """
        max_iters = 100
        
        for _ in range(max_iters):
            continue_reduction = self.reduce_state_single_step()
            if not continue_reduction:
                break
    
    def reduce_state_single_step(self) -> bool:
        """
        Perform one step of game board state reduction.
        Returns `true` if more reduction should be completed,
        `false` otherwise.
        """
        
        # If no moves have been made, then uncover a tile in
        # the middle of the board
        if not self.__any_moves_made:
            self.set_tile(self.cols / 2, self.rows / 2, "UNCOVER")
        
        return False

class GameGridInteractionLayer:
    def __init__(self, grid: GameGridCalibration):
        self.grid = grid
        self.cols = grid.num_tile_cols
        self.rows = grid.num_tile_rows
        self.__true_game_state = MinesweeperGameState(self.cols, self.rows)
        self.__last_game_screenshot: Image.Image|None = None
    
    def __get_new_game_screenshot(self) -> Image.Image:
        pyautogui.moveTo(self.grid.flag_icon_position.x, self.grid.flag_icon_position.y)
        time.sleep(0.8)
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
        content_x_start = None
        content_x_end = None
        
        y = int(round((y1 - y0) / 2 + y0))
        for x in range(x0, x1):
            
            # Extract the pixel color from the screenshot
            pixel_color = Color.from_pixel(self.__last_game_screenshot, x, y)
            
            # Get the base color of the tile if it hasn't been found
            if base_color == None:
                base_color = pixel_color
            
            # Record the start and end of the cross-section of the content
            # to later find the color of its center
            if pixel_color != base_color and content_x_start is None:
                # Non-base color pixel and content_x_start not yet found -- set it
                content_x_start = x
            if pixel_color == base_color and content_x_start is not None:
                # Base color pixel and content_x_start found -- end has been found
                content_x_end = x
                break
        
        # The color of the foreground, if a foreground exists
        content_color = None
        
        # If a foreground exists
        if content_x_start is not None:
            
            # Set end X of foreground if one wasn't found
            if content_x_end is None:
                content_x_end = x1
            
            # Get the center of the foreground
            content_x_center = (content_x_end - content_x_start) / 2 + content_x_start
            content_color = Color.from_pixel(self.__last_game_screenshot, content_x_center, y)
        
        return self.__tile_colors_lookup(base_color, content_color)
    
    def __tile_colors_lookup(self, base_color: Color, content_color: Color|None) -> TILE_STATE:
        if base_color == self.grid.tileColor1 or base_color == self.grid.tileColor2:
            # Green tile background
            if content_color == None:
                return "GREEN"
            else:
                # We can just assume the 
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
                    Color(0x1976D2), # 1
                    Color(0x388E3C), # 2
                    Color(0xD32F2F), # 3
                    Color(0x7B1FA2), # 4
                    Color(0xFF8F00), # 5
                    Color(0x0097A7), # 6
                    Color(0x424242), # 7
                    Color(0x9E9C9A), # 8
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
        for col in range(self.cols):
            for row in range(self.rows):
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
            return self.__tile_lookup(col, row)
    
    def __click_tile(self, col: int, row: int, button):
        location = self.grid.get_position(col, row)
        pyautogui.click(location.x, location.y, button=button)
    
    def uncover_tile(self, col: int, row: int):
        """
        Uncover (left click) a given tile. Fails if the tile is not GREEN or MINE
        (i.e. if it's not an actionable tile).
        """
        
        # Check if we can uncover a mine here
        tile_state = self.get_tile(col, row)
        if tile_state != "GREEN" and tile_state != "MINE":
            raise MinesweeperException(f"Cannot uncover non-green or non-flag tile: {col} x {row}")
        
        # Right click the tile if it has a flag on it to remove the flag first
        if tile_state == "MINE":
            self.__click_tile(col, row, "RIGHT")
        
        # Click the tile to uncover
        self.__click_tile(col, row, "LEFT")
        
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
            # Update the game state
            self.__true_game_state.set_tile(col, row, "MINE" if flagged_status else "GREEN")

class GameSolver:
    def __init__(self, interact: GameGridInteractionLayer):
        self.interact = interact
        self.game_state = MinesweeperGameState(self.interact.cols, self.interact.rows)
    
    def solve(self):
        self.__solve_cycle()
    
    def __solve_cycle(self):
        self.interact.uncover_tile(0, 0)
        
        for col in range(self.interact.cols):
            for row in range(self.interact.rows):
                tile_state = self.interact.get_tile(col, row)
                if isinstance(tile_state, int):
                    print(tile_state)
                    pyautogui.moveTo(self.interact.grid.get_position(col, row, Vec2D(0, 0)).components())
                    pyautogui.moveTo(self.interact.grid.get_position(col, row, Vec2D(1, 0)).components(), duration=0.25)
                    pyautogui.moveTo(self.interact.grid.get_position(col, row, Vec2D(1, 1)).components(), duration=0.25)
                    pyautogui.moveTo(self.interact.grid.get_position(col, row, Vec2D(0, 1)).components(), duration=0.25)
                    pyautogui.moveTo(self.interact.grid.get_position(col, row, Vec2D(0, 0)).components(), duration=0.25)
        pass

def sweep_mines(debug: bool) -> bool:
    print("== Minesweeper Solver ==")
    try:
        grid = GameGridCalibration(debug)
        interact = GameGridInteractionLayer(grid)
        solver = GameSolver(interact)
        
        solver.solve()
        
        # # TODO: Add time to complete from first click
        
        # game_state = MinesweeperGameState(grid.num_tile_cols, grid.num_tile_rows)
        # game_state.reduce_state()
        
        
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
    time.sleep(3)
    
    try:
        sweep_mines(debug)
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
