import pygame
import random
import math
from typing import List
import sys

pygame.init()
random.seed(42)
FRAME_RATE = 60
WORLD_SIZE = 3000
SCREEN_WIDTH, SCREEN_HEIGHT = (1000, 700)
WALL_DELTA = 20
INITIAL_ZOOM = 10
SURFACE_COLOR = (242, 251, 255)
GRID_COLOR = (230, 240, 240)
NUM_CELLS = WORLD_SIZE ** 2 // int(1e3)
START_MASS = 3
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


class Camera:
    def __init__(self):
        self.x, self.y = 0, 0
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.zoom = INITIAL_ZOOM

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


class Cell:
    def __init__(self, surface, x, y, mass, color):
        self.x = x
        self.y = y
        self.mass = mass
        self.surface = surface
        self.color = color

    @classmethod
    def new_random_cell(cls, surface):
        x = random.randint(WALL_DELTA, WORLD_SIZE - WALL_DELTA)
        y = random.randint(WALL_DELTA, WORLD_SIZE - WALL_DELTA)
        mass = random.randint(MIN_CELL_MASS, MAX_CELL_MASS)
        color = random.choice(CELL_COLORS)
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


class Player(Cell):
    def __init__(self, surface, x, y, mass, color, speed):
        super().__init__(surface, x, y, mass, color)
        self.speed = speed

    @property
    def score(self):
        return int(self.mass)

    def update(self, action, game_state):
        angle = action
        cells = game_state
        self.move(angle)
        self.detect_collisions(cells)
        self.update_speed()

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

class AgarioGame:
    def __init__(self):
        self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Agar.io")
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.cells = []
        self.camera = Camera()
        self.player = Player(
            surface=self.surface,
            x=WORLD_SIZE // 2, y=WORLD_SIZE // 2,
            mass=START_MASS, color=random.choice(PLAYER_COLORS),
            speed=MAX_PLAYER_SPEED
        )
        self.create_random_cells(NUM_CELLS)

    def create_random_cells(self, num_cells) -> None:
        self.cells = [Cell.new_random_cell(self.surface) for _ in range(num_cells)]

    def draw_grid(self):
        for i in range(0, WORLD_SIZE + 1, WORLD_SIZE // 100):
            horizontal_start = self.camera.transform_pos(0, i)
            horizontal_end = self.camera.transform_pos(WORLD_SIZE, i)
            vertical_start = self.camera.transform_pos(i, 0)
            vertical_end = self.camera.transform_pos(i, WORLD_SIZE)
            pygame.draw.line(self.surface, GRID_COLOR, horizontal_start, horizontal_end)
            pygame.draw.line(self.surface, GRID_COLOR, vertical_start, vertical_end)

    def get_next_action(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        x_centered = mouse_x - SCREEN_WIDTH / 2
        y_centered = mouse_y - SCREEN_HEIGHT / 2
        angle = -math.atan2(y_centered, x_centered)
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
        self.camera.update(self.player)

    def render(self):
        self.surface.fill(SURFACE_COLOR)
        self.draw_grid()
        self.player.draw(self.camera)
        for cell in self.cells:
            cell.draw(self.camera)
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
