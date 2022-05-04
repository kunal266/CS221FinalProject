from __future__ import annotations  # for type hints of the same type as defined class

import pygame
import random
import math
from typing import List, Tuple
import sys

pygame.init()

GENERAL_SCALE = 1
CAMERA_SHOULD_FOLLOW_PLAYER = True

WORLD_SIZE = 1000
SCREEN_WIDTH, SCREEN_HEIGHT = (1000, 1000)
LEADERBOARD_SHAPE = (SCREEN_WIDTH // 7, SCREEN_HEIGHT // 2)
GRID_STEP = WORLD_SIZE // 100
WALL_DELTA = 20
INITIAL_ZOOM = 9
SURFACE_COLOR = (242, 251, 255)
GRID_COLOR = (230, 240, 240)
# NUM_CELLS = WORLD_SIZE ** 2 // int(1e3)
NUM_CELLS = 3*WORLD_SIZE ** 2 // int(1e3)
# NUM_ADVERSARIES = 14
NUM_ADVERSARIES = 0
START_MASS = 10
MIN_CELL_MASS = 1
MAX_CELL_MASS = 2
MAX_PLAYER_SPEED = 1
CAMERA_ZOOM_UPDATE_SPEED = 0.015
RADIUS_MASS_EXPONENT = 0.8
SPEED_UPDATE_SPEED = 0.01
MASS_UPDATE_SPEED = 0.05
PELLET_MASS_DISCOUNT = 0.5
PLAYER_MASS_DISCOUNT = 1.0
PLAYER_COLORS = [(37, 7, 255), (35, 183, 253), (48, 254, 241), (19, 79, 251), (255, 7, 230), (255, 7, 23),
                 (6, 254, 13)]
CELL_COLORS = [(80, 252, 54), (36, 244, 255), (243, 31, 46), (4, 39, 243), (254, 6, 178), (255, 211, 7), (216, 6, 254),
               (145, 255, 7), (7, 255, 182), (255, 6, 86), (147, 7, 255)]
COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
}
FONT_SIZE = 24

# set these on AgarioGame's initialization, not here
FONT = None
DISPLAY_TEXT = None

if GENERAL_SCALE != 1:
    GRID_STEP = int(GRID_STEP * GENERAL_SCALE)
    START_MASS = int(START_MASS * GENERAL_SCALE)
    MIN_CELL_MASS = int(MIN_CELL_MASS * GENERAL_SCALE)
    MAX_CELL_MASS = int(MAX_CELL_MASS * GENERAL_SCALE)
    FONT_SIZE = int(FONT_SIZE * GENERAL_SCALE)
    NUM_CELLS = int(NUM_CELLS / GENERAL_SCALE)
    WALL_DELTA = int(WALL_DELTA / GENERAL_SCALE)


def SCALE_ALL_SPEEDS(speed_scale: float) -> None:
    global MAX_PLAYER_SPEED
    MAX_PLAYER_SPEED *= speed_scale


def DRAW_TEXT(surface, text, position, color=COLORS['black']) -> None:
    surface.blit(FONT.render(text, True, color), position)


class Camera:
    def __init__(self):
        self.x, self.y = 0, 0
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.zoom = INITIAL_ZOOM

    def __str__(self):
        return f'Camera -> pos=({self.x}, {self.y}), width={self.width}, ' \
               f'height={self.height}, zoom={self.zoom}'

    def update(self, player):
        self.update_zoom(player)
        self.centre(player)

    def update_zoom(self, player):
        self.zoom = (INITIAL_ZOOM - 1) * math.exp(-CAMERA_ZOOM_UPDATE_SPEED * (player.mass - MIN_CELL_MASS)) + 1

    def centre(self, player):
        self.x = player.x - (1 / self.zoom) * (self.width / 2)
        self.y = player.y - (1 / self.zoom) * (self.height / 2)

    def transform_pos(self, x, y):
        relative_x = int((x - self.x) * self.zoom)
        relative_y = int((y - self.y) * self.zoom)
        return relative_x, relative_y

    def transform_pos_inv(self, rel_x, rel_y):
        absolute_x = int((rel_x / self.zoom) + self.x)
        absolute_y = int((rel_y / self.zoom) + self.y)
        return absolute_x, absolute_y

    def transform_and_center_pos(self, x, y):
        rel_x, rel_y = self.transform_pos(x, y)
        centered_x = int(rel_x - SCREEN_WIDTH / 2)
        centered_y = int(rel_y - SCREEN_HEIGHT / 2)
        return centered_x, centered_y

    def is_visible(self, cell):
        x, y = self.transform_pos(cell.x, cell.y)
        return 0 < x < SCREEN_WIDTH and 0 < y < SCREEN_HEIGHT


class Cell:
    def __init__(self, surface, x, y, mass, color):
        self.x = x
        self.y = y
        self.mass = mass
        self.surface = surface
        self.color = color
        self.is_alive = True

    def __str__(self):
        return f'pos=({self.x}, {self.y}), mass={self.mass}'

    @staticmethod
    def get_random_properties(min_mass, max_mass):
        x = random.randint(WALL_DELTA, WORLD_SIZE - WALL_DELTA)
        y = random.randint(WALL_DELTA, WORLD_SIZE - WALL_DELTA)
        mass = random.randint(min_mass, max_mass)
        color = random.choice(CELL_COLORS)
        return x, y, mass, color

    @classmethod
    def new_random_cell(cls, surface):
        x, y, mass, color = cls.get_random_properties(MIN_CELL_MASS, MAX_CELL_MASS)
        return cls(surface, x, y, mass, color)

    @property
    def radius(self):
        return int(self.mass) ** RADIUS_MASS_EXPONENT

    def draw(self, cam):
        x, y = cam.transform_pos(self.x, self.y)
        radius = int(self.radius * cam.zoom)
        dimmed_color = tuple(x // 2 for x in self.color)
        pygame.draw.circle(self.surface, dimmed_color, (x, y), radius)
        pygame.draw.circle(self.surface, self.color, (x, y), max(radius * 0.7, radius - 10))

    @staticmethod
    def distance(cell1, cell2) -> float:
        return math.sqrt(math.fabs(cell1.x - cell2.x) ** 2 + math.fabs(cell1.y - cell2.y) ** 2)

    @staticmethod
    def scaled_distance(cell1, cell2, camera) -> float:
        x1, y1 = camera.transform_pos(cell1.x, cell2.y)
        x2, y2 = camera.transform_pos(cell2.x, cell2.y)
        return math.sqrt(math.fabs(x1 - x2) ** 2 + math.fabs(y1 - y2) ** 2)


class Player(Cell):
    def __init__(self, surface, x, y, mass, color, name):
        super().__init__(surface, x, y, mass, color)
        self.speed = MAX_PLAYER_SPEED
        self.camera = Camera()
        self.camera.update(self)
        self.name = name

    @property
    def score(self):
        return int(self.mass)

    def draw_player(self, cam):
        self.draw(cam)
        if DISPLAY_TEXT:
            name_width, name_height = FONT.size(self.name)
            score_width, score_height = FONT.size(str(self.score))
            x_draw, y_draw = cam.transform_pos(self.x, self.y)
            DRAW_TEXT(self.surface, text=self.name,
                      position=(x_draw - name_width / 2, y_draw - name_height),
                      color=COLORS['black'])
            DRAW_TEXT(self.surface, text=str(self.score),
                      position=(x_draw - score_width / 2, y_draw + score_height),
                      color=COLORS['black'])

    def update_from_action(self, action, game_state):
        angle = action
        cells, adversaries, player = game_state
        self.move(angle)
        self.detect_collisions(cells, adversaries, player)
        self.update_speed()
        self.camera.update(self)

    def update_speed(self):
        self.speed = MAX_PLAYER_SPEED * math.exp(-SPEED_UPDATE_SPEED * (self.mass - MIN_CELL_MASS))

    def detect_collisions(self, cells, adversaries, player):
        all_cells = cells + adversaries + [player]
        for cell in all_cells:
            if cell is not self and cell.mass < self.mass:
                if Cell.distance(cell, self) < self.radius:
                    self.add_mass(cell)
                    cell.is_alive = False

    def add_mass(self, cell):
        extra_mass = cell.mass * math.exp(-MASS_UPDATE_SPEED * (self.mass - MIN_CELL_MASS))
        if isinstance(cell, Player):
            self.mass += extra_mass * PLAYER_MASS_DISCOUNT
        else:
            self.mass += extra_mass * PELLET_MASS_DISCOUNT

    def move(self, angle):
        vx = self.speed * math.cos(angle)
        vy = - self.speed * math.sin(angle)
        self.x += vx
        self.y += vy
        within_x_min, within_x_max, within_y_min, within_y_max = self.is_within_boundaries()
        if not within_x_min:
            self.x = 0 + self.radius
        if not within_x_max:
            self.x = WORLD_SIZE - self.radius
        if not within_y_min:
            self.y = 0 + self.radius
        if not within_y_max:
            self.y = WORLD_SIZE - self.radius

    def is_within_boundaries(self):
        x, y = self.x, self.y
        radius = self.radius
        within_x_min = 0 <= x - radius
        within_x_max = WORLD_SIZE >= x + radius
        within_y_min = 0 <= y - radius
        within_y_max = WORLD_SIZE >= y + radius
        return within_x_min, within_x_max, within_y_min, within_y_max

    @classmethod
    def new_random_player(cls, surface, name):
        x, y, mass, color = cls.get_random_properties(START_MASS, START_MASS)
        return cls(surface, x, y, mass, color, name)

    def get_visible_game_state(self, game_state) -> Tuple[List[Cell], List[Player], Player]:
        cells, adversaries, player = game_state
        all_players = adversaries + [player]
        visible_cells = list(filter(lambda cell: self.camera.is_visible(cell), cells))
        visible_adversaries = list(filter(lambda cell: cell is not self and self.camera.is_visible(cell), all_players))
        return visible_cells, visible_adversaries, self

    def __str__(self):
        return f'{self.name}: {self.score}'

    def __repr__(self):
        return self.__str__()


class MainPlayer(Player):
    def __init__(self, surface, x, y, mass, color, name):
        super().__init__(surface, x, y, mass, color, name)

    @classmethod
    def new_random_main_player(cls, surface):
        return cls.new_random_player(surface, name='Player')

    def update(self, action, game_state):
        self.update_from_action(action, game_state)


class GreedyAdversary(Player):
    # static property
    static_id = 0

    def __init__(self, surface, x, y, mass, color, name):
        super().__init__(surface, x, y, mass, color, name)
        self.last_angle = 0

    @classmethod
    def new_random_greedy_adversary(cls, surface):
        name = f'Blob {cls.static_id}'
        cls.static_id += 1
        return cls.new_random_player(surface, name=name)

    def choose_action(self, visible_game_state):
        cells, players, _ = visible_game_state
        all_cells = cells + players
        all_smaller_cells = list(filter(lambda cell: cell.mass < self.mass, all_cells))
        if not all_smaller_cells:
            return self.last_angle
        best_cell = max(all_smaller_cells, key=lambda cell: cell.mass / (Cell.distance(cell, self) + 0.001))
        best_x, best_y = self.camera.transform_and_center_pos(best_cell.x, best_cell.y)
        angle = -math.atan2(best_y, best_x)
        return angle

    def update(self, game_state):
        visible_game_state = self.get_visible_game_state(game_state)
        action = self.choose_action(visible_game_state)
        self.update_from_action(action, game_state)
        self.last_angle = action


class AgarioGame:
    def __init__(self,
                 should_render: bool,
                 speed_scale: float,
                 display_text: bool,
                 should_display: bool):
        self.surface = None
        self.clock = None
        self.player = None
        self.adversaries = None
        self.cells = None
        self.leaderboard_surface = None
        self.camera = None
        self.game_ended = None
        self.should_render = should_render
        self.should_display = should_display
        global FONT, DISPLAY_TEXT
        SCALE_ALL_SPEEDS(speed_scale)
        if should_display and not self.should_render:
            raise Exception('should_display can only be true when should_render is')
        if display_text:
            if not should_render and should_display:
                raise Exception('display_text can only be set when should_render and should_display are true')
            DISPLAY_TEXT = display_text
            FONT = pygame.font.SysFont(None, FONT_SIZE)
        self.reset()

    def reset(self):
        self.game_ended = False
        if self.should_render:
            self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            # pygame.display.set_caption("Agar.io")
            if DISPLAY_TEXT:
                self.leaderboard_surface = pygame.Surface(LEADERBOARD_SHAPE, pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        self.cells: List[Cell] = []
        self.adversaries: List[GreedyAdversary] = []
        self.player = MainPlayer.new_random_main_player(self.surface)
        self.cells = self.create_random_cells(NUM_CELLS)
        self.adversaries = self.create_random_adversaries(NUM_ADVERSARIES)
        if CAMERA_SHOULD_FOLLOW_PLAYER:
            self.camera = self.player.camera
        else:
            self.camera = Camera()

    @staticmethod
    def seed(seed):
        random.seed(seed)

    def respawn_player(self):
        self.player = MainPlayer.new_random_main_player(self.surface)
        if CAMERA_SHOULD_FOLLOW_PLAYER:
            self.camera = self.player.camera

    @property
    def game_state(self):
        return self.cells, self.adversaries, self.player

    def get_player_state(self):
        return self.player.get_visible_game_state(self.game_state)

    def create_random_cells(self, num_cells) -> List[Cell]:
        return [Cell.new_random_cell(self.surface) for _ in range(num_cells)]

    def create_random_adversaries(self, num_adversaries) -> List[GreedyAdversary]:
        return [GreedyAdversary.new_random_greedy_adversary(self.surface) for _ in range(num_adversaries)]

    def draw_grid(self):
        for i in range(0, WORLD_SIZE + 1, GRID_STEP):
            horizontal_start = self.camera.transform_pos(0, i)
            horizontal_end = self.camera.transform_pos(WORLD_SIZE, i)
            vertical_start = self.camera.transform_pos(i, 0)
            vertical_end = self.camera.transform_pos(i, WORLD_SIZE)
            pygame.draw.line(self.surface, GRID_COLOR, horizontal_start, horizontal_end)
            pygame.draw.line(self.surface, GRID_COLOR, vertical_start, vertical_end)

    def get_next_action(self):
        if not self.should_render:
            return 0
        mouse_x, mouse_y = pygame.mouse.get_pos()
        abs_mouse_x, abs_mouse_y = self.camera.transform_pos_inv(mouse_x, mouse_y)
        mouse_x_rel_player, mouse_y_rel_player = self.player.camera.transform_and_center_pos(abs_mouse_x, abs_mouse_y)
        angle = -math.atan2(mouse_y_rel_player, mouse_x_rel_player)
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        return angle

    @staticmethod
    def filter_alive(cells):
        return list(filter(lambda cell: cell.is_alive, cells))

    def spawn_new_cells_and_adversaries(self):
        if len(self.cells) < NUM_CELLS:
            self.cells += self.create_random_cells(NUM_CELLS - len(self.cells))
        if len(self.adversaries) < NUM_ADVERSARIES:
            self.adversaries += self.create_random_adversaries(NUM_ADVERSARIES - len(self.adversaries))

    def close(self):
        pygame.quit()

    def step(self, action):
        if self.game_ended:
            # pygame.quit()
            # sys.exit()
            return
        self.player.update(action, self.game_state)
        for adversary in self.adversaries:
            adversary.update(self.game_state)
        self.cells = self.filter_alive(self.cells)
        self.adversaries = self.filter_alive(self.adversaries)
        self.spawn_new_cells_and_adversaries()
        if not self.player.is_alive:
            self.game_ended = True
            # self.respawn_player()

    def render(self):
        assert self.should_render
        self.surface.fill(SURFACE_COLOR)
        self.draw_grid()
        self.player.draw_player(self.camera)
        for cell in self.cells:
            cell.draw(self.camera)
        for adversary in self.adversaries:
            adversary.draw_player(self.camera)
        if DISPLAY_TEXT:
            self.draw_leaderboard()
        if self.should_display:
            pygame.display.flip()
        image_array = pygame.surfarray.array3d(self.surface)
        return image_array

    def get_sorted_players(self):
        all_players = self.adversaries + [self.player]
        all_players.sort(key=lambda x: x.score, reverse=True)
        return all_players

    def draw_leaderboard(self):
        # Draw score in lower-left corner
        DRAW_TEXT(self.surface, text=f'Score: {self.player.score}',
                  position=(FONT_SIZE, SCREEN_HEIGHT - FONT_SIZE),
                  color=(0, 0, 0))
        # Draw leaderboard
        width, height = LEADERBOARD_SHAPE
        pos_x, pos_y = SCREEN_WIDTH - width, SCREEN_HEIGHT // 20
        self.surface.blit(self.leaderboard_surface, (pos_x, pos_y))
        # Add leading scores
        all_players = self.get_sorted_players()
        top_10 = all_players[:10]
        for i, player in enumerate(top_10):
            color = COLORS['red'] if player is self.player else COLORS['black']
            DRAW_TEXT(self.surface, text=f'{i + 1}. {player.name}',
                      position=(pos_x + 3, pos_y + (FONT_SIZE + 3) * i), color=color)
        if self.player not in top_10:
            rank = all_players.index(self.player) + 1
            DRAW_TEXT(self.surface, text=f'',
                      position=(pos_x + 3, pos_y + (FONT_SIZE + 3) * len(top_10)))
            DRAW_TEXT(self.surface, text=f'{rank}. {self.player.name}',
                      position=(pos_x + 3, pos_y + (FONT_SIZE + 3) * (len(top_10) + 1)),
                      color=COLORS['red'])

    def run(self):
        i = 0
        miliseconds = 0
        while True:
            action = self.get_next_action()
            self.step(action)
            if self.should_render:
                self.render()
            else:
                print(self.get_sorted_players()[:5])
            miliseconds += self.clock.tick()
            i += 1
            if miliseconds >= 1000:
                # print(f'fps = {i}')
                i = miliseconds = 0


if __name__ == '__main__':
    game = AgarioGame(should_render=True,
                      speed_scale=2,
                      display_text=True,
                      should_display=True)
    game.run()
