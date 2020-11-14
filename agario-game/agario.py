import pygame
import random
import math
from typing import List
import sys

pygame.init()

GENERAL_SCALE = 10

random.seed(42)
FRAME_RATE = 60
WORLD_SIZE = 1000
SCREEN_WIDTH, SCREEN_HEIGHT = (1000, 1000)
GRID_STEP = WORLD_SIZE // 100
# WORLD_SIZE = 500
# SCREEN_WIDTH, SCREEN_HEIGHT = (500, 500)
WALL_DELTA = 20
# INITIAL_ZOOM = 10
INITIAL_ZOOM = 1
SURFACE_COLOR = (242, 251, 255)
GRID_COLOR = (230, 240, 240)
NUM_CELLS = WORLD_SIZE ** 2 // int(1e3)
NUM_ADVERSARIES = 10
START_MASS = 5
MIN_CELL_MASS = 1
MAX_CELL_MASS = 2
MAX_PLAYER_SPEED = 1
CAMERA_ZOOM_UPDATE_SPEED = 0.025
RADIUS_MASS_EXPONENT = 0.7
SPEED_UPDATE_SPEED = 0.005
MASS_UPDATE_SPEED = 0.05
MASS_DISCOUNT = 0.5
PLAYER_COLORS = [(37, 7, 255), (35, 183, 253), (48, 254, 241), (19, 79, 251), (255, 7, 230), (255, 7, 23),
                 (6, 254, 13)]
CELL_COLORS = [(80, 252, 54), (36, 244, 255), (243, 31, 46), (4, 39, 243), (254, 6, 178), (255, 211, 7), (216, 6, 254),
               (145, 255, 7), (7, 255, 182), (255, 6, 86), (147, 7, 255)]
FONT_SIZE = 24
DISPLAY_TEXT = False

if GENERAL_SCALE != 1:
    GRID_STEP = int(GRID_STEP * GENERAL_SCALE)
    START_MASS = int(START_MASS * GENERAL_SCALE)
    MIN_CELL_MASS = int(MIN_CELL_MASS * GENERAL_SCALE)
    MAX_CELL_MASS = int(MAX_CELL_MASS * GENERAL_SCALE)
    FONT_SIZE = int(FONT_SIZE * GENERAL_SCALE)
    NUM_CELLS = int(NUM_CELLS / GENERAL_SCALE)
    WALL_DELTA = int(WALL_DELTA / GENERAL_SCALE)
    # MAX_PLAYER_SPEED = MAX_PLAYER_SPEED * GENERAL_SCALE


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
    def __init__(self, surface, x, y, mass, color):
        super().__init__(surface, x, y, mass, color)
        self.speed = MAX_PLAYER_SPEED
        self.camera = Camera()
        self.camera.update(self)

    @property
    def score(self):
        return int(self.mass)

    def update_from_action(self, action, game_state):
        angle = action
        cells = game_state
        self.move(angle)
        self.detect_collisions(cells)
        self.update_speed()
        self.camera.update(self)

    def update_speed(self):
        self.speed = MAX_PLAYER_SPEED * math.exp(-SPEED_UPDATE_SPEED * (self.mass - MIN_CELL_MASS))

    def detect_collisions(self, cells):
        for cell in cells:
            if cell.mass < self.mass:
                if Cell.distance(cell, self) < self.radius:
                    self.add_mass(cell.mass)
                    cells.remove(cell)
                    cells.append(Cell.new_random_cell(self.surface))

    def add_mass(self, mass):
        extra_mass = mass * math.exp(-MASS_UPDATE_SPEED * (self.mass - MIN_CELL_MASS))
        self.mass += extra_mass * MASS_DISCOUNT

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


class MainPlayer(Player):
    def __init__(self, surface, x, y, mass, color):
        super().__init__(surface, x, y, mass, color)

    def update(self, action, game_state):
        self.update_from_action(action, game_state)


class GreedyAdversary(Player):
    def __init__(self, surface, x, y, mass, color):
        super().__init__(surface, x, y, mass, color)
        self.last_angle = 0

    @classmethod
    def new_random_greedy_adversary(cls, surface):
        x, y, mass, color = cls.get_random_properties(START_MASS, START_MASS)
        return cls(surface, x, y, mass, color)

    def get_visible_game_state(self, game_state):
        cells = game_state
        visible_cells = list(filter(lambda cell: self.camera.is_visible(cell), cells))
        return visible_cells

    def update(self, game_state):
        visible_game_state = self.get_visible_game_state(game_state)
        action = self.choose_action(visible_game_state)
        self.update_from_action(action, game_state)
        self.last_angle = action

    def choose_action(self, visible_game_state):
        cells = visible_game_state
        if not cells:
            return self.last_angle
        best_cell = max(cells, key=lambda cell: cell.mass / Cell.distance(cell, self))
        best_x, best_y = self.camera.transform_and_center_pos(best_cell.x, best_cell.y)
        angle = -math.atan2(best_y, best_x)
        return angle


class AgarioGame:
    def __init__(self):
        self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Agar.io")
        if DISPLAY_TEXT:
            self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.cells: List[Cell] = []
        self.adversaries: List[GreedyAdversary] = []
        self.player = MainPlayer(
            surface=self.surface,
            x=WORLD_SIZE // 2, y=WORLD_SIZE // 2,
            mass=START_MASS, color=random.choice(PLAYER_COLORS)
        )
        self.create_random_cells(NUM_CELLS)
        self.create_random_adversaries(NUM_ADVERSARIES)
        self.camera = Camera()

    def create_random_cells(self, num_cells) -> None:
        self.cells = [Cell.new_random_cell(self.surface) for _ in range(num_cells)]

    def create_random_adversaries(self, num_adversaries):
        self.adversaries = [GreedyAdversary.new_random_greedy_adversary(self.surface) for _ in range(num_adversaries)]

    def draw_grid(self):
        for i in range(0, WORLD_SIZE + 1, GRID_STEP):
            horizontal_start = self.camera.transform_pos(0, i)
            horizontal_end = self.camera.transform_pos(WORLD_SIZE, i)
            vertical_start = self.camera.transform_pos(i, 0)
            vertical_end = self.camera.transform_pos(i, WORLD_SIZE)
            pygame.draw.line(self.surface, GRID_COLOR, horizontal_start, horizontal_end)
            pygame.draw.line(self.surface, GRID_COLOR, vertical_start, vertical_end)

    def get_next_action(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_x_rel, mouse_y_rel = self.player.camera.transform_and_center_pos(mouse_x, mouse_y)
        angle = -math.atan2(mouse_y_rel, mouse_x_rel)
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        return angle

    def step(self, action):
        self.player.update(action, self.cells)
        for adversary in self.adversaries:
            adversary.update(self.cells)

    def render(self):
        self.surface.fill(SURFACE_COLOR)
        self.draw_grid()
        self.player.draw(self.camera)
        for cell in self.cells:
            cell.draw(self.camera)
        for adversary in self.adversaries:
            adversary.draw(self.camera)
        if DISPLAY_TEXT:
            score_text = self.font.render(f'Score: {self.player.score}', True, (0, 0, 0))
            self.surface.blit(score_text, (FONT_SIZE, SCREEN_HEIGHT - FONT_SIZE))
        pygame.display.flip()

    def run(self):
        while True:
            action = self.get_next_action()
            self.step(action)
            self.render()
            self.clock.tick(FRAME_RATE)


if __name__ == '__main__':
    game = AgarioGame()
    game.run()
