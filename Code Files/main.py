# Uchinogasu Main File

# Import and initialisation statements
import pygame
import sys
import random
import numpy as np
import pandas as pd
import gc
import datastuffs

from os.path import join
from PIL import ImageColor
from sklearn.linear_model import LinearRegression, LogisticRegression

"""
Colours from PIL:
https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/_static/exams/result/ImageColor_01.html
"""

pygame.init()

# General use functions and procedures

def get_sprite_sheet(path: str, frame: tuple):
    """Returns part of a sprite sheet."""
    image = pygame.image.load(path).convert_alpha() # Get the image
    desired_frame = image.subsurface(pygame.Rect(frame)) # Create a surface from the image using the frame (x, y, width, height)
    return desired_frame

# Game Classes
class Initialisation:
    score = 0
    def __init__(self):
        """Initialization class"""
        # Init statements / setup
        self._width = 1280
        self._height = 720
        self._screen = pygame.display.set_mode((self._width, self._height))
        self._transparency_layer = pygame.Surface((self._width, self._height), pygame.SRCALPHA)
        pygame.display.set_caption("Uchinogasu")
        pygame.display.set_icon(pygame.image.load(join("Assets", "Images", "uchinogasu.png")).convert_alpha())
        self._clock = pygame.time.Clock()
        self._FPS = 60
        self._BLOCK_SIZE = 100

        # Colours
        self._WHITE = ImageColor.getrgb('white')
        self._BLUE = (24, 48, 72)
        self._GREY = [val - 55 for val in ImageColor.getrgb('dimgrey')]
        self._GREYER = [val - 85 for val in ImageColor.getrgb('dimgrey')]
        self._BLACK = ImageColor.getrgb('black')
        self._ALPHA = (255, 255, 255, 0)

        # Backgrounds
        self._main_game_bg = [join("Assets", "Backgrounds", f"space_bg_{n}.png") for n in range(1, 6)]
        self._loss_bg = [join("Assets", "Backgrounds", f"space_bg_loss_{n}.png") for n in range(1, 6)]

        # Dataframes
        self.highscores_df = pd.read_csv(join('Data', 'highscores.csv'))

class Uchinogasu(Initialisation):
    # Game classes
    def __init__(self):
        """A game about dragging a block to the right spot."""
        super().__init__()

        # Assets
        # Text
        # self._title = Text("Uchinogasu", self._width // 2, self._height // 6, 100, self._WHITE)
        self._scoretxt = Text(f"Score: {Initialisation.score}", 60, 20, 30, self._WHITE, font_choice=1)
        self._tutorial_text = Text("HOW TO PLAY: CHOOSE BLOCK, COMPUTER FIRES UP, DON'T CHOOSE WHAT COMPUTER CHOOSES",
                                   -3125, self._height - 50, 50, self._WHITE, font_choice=0)
        self.disable_tutorial_text = False

        # Backgrounds / Overlay
        self.background1 = Background(random.choice(self._main_game_bg), self._width, self._height)
        self.background2 = Background(random.choice(self._loss_bg), self._width, self._height, direction=1)
        self._fade = False
        
        # Objects
        self.box1 = Box(self._width//4.5, self._height//2.5, join('Assets', 'Tiles', 'b1.png'), (64, 64, 64, 64))
        self.box2 = Box(self._width//4.5 + 200, self._height//2.5, join('Assets', 'Tiles', 'b2.png'), (64, 64, 64, 64))
        self.box3 = Box(self._width//4.5 + 400, self._height//2.5, join('Assets', 'Tiles', 'b3.png'), (64, 64, 64, 64))
        self.box4 = Box(self._width //4.5 + 600, self._height//2.5, join('Assets', 'Tiles', 'b4.png'), (64, 64, 64, 64))

        # Losing Menu
        self._loss_scoretxt = Text(f"Score: {Initialisation.score}", self._width//2, self._height//2.5, 50, self._WHITE, font_choice=0)
        self._losstxt = Text(f"You Lose!", self._width//2, self._height//2, 100, self._WHITE)
        self._instruct_on_losstxt = Text(f"Press R to restart", self._width//4, self._height//1.45, 25, self._WHITE)
        

        self._instruct_on_losstxt_2 = Text(f"Press T to toggle tutorial text [ON]", 
                                           self._width//1.5, self._height//1.45, 25, self._WHITE)
        self.r_key = AnimatedImage(join("Assets", "Images", "r_key.png"), 17, 16, (0,0), self._width//4, self._height//1.25, 128, 2, (1/30))
        self.t_key = AnimatedImage(join("Assets", "Images", "t_key.png"), 17, 16, (0,0), self._width//1.5, self._height//1.25, 128, 2, (1/30))
        self.overlay = Overlay(self._width, self._height, 0, 255, 80, self._BLACK, "fadein")

        self._highscore_txt = Text(f"Top Scores: {self.highscores_df.iat[0, 0]}, {self.highscores_df.iat[1, 0]}, {self.highscores_df.iat[2, 0]}", self._width//2,
                                    self._height//1.75, 30, self._WHITE)
    
        # Groups
        self.boxes = pygame.sprite.Group()
        self.boxes.add(self.box1, self.box2, self.box3, self.box4)

        # Regressor Stuff
        self._mode = np.random.randint(0, 2, 1)[0]
        self.regressor = Regressor(mode=self._mode)
        self.regressor_group = pygame.sprite.Group()
        self.regressor_group.add(self.regressor.regressor_warning)
        self.loss = False
        self.readyred = False

        # Creating and editing files
        datastuffs.create_choice_file() # Box choices
        datastuffs.create_high_score_file() # High scores

        # Only for Logistic regression, so that 'binary search'
        if self._mode == 2:
            datastuffs.inflate_choice_file()

    def reset_stats(self):
        """Resets the statistics of the game so that a game can be repeated."""

        # Initialisation stuffs
        self.loss = False
        self.readyred = False

        Initialisation.score = 0
        self._scoretxt = Text(f"Score: {Initialisation.score}", 60, 20, 30, self._WHITE, font_choice=1)
        datastuffs.create_choice_file()

        self._tutorial_text._rect.x = -3125

        # The backgrounds
        self.background1.set_speed(1)
        Box.bg_speed_ref = 1
        self.background1.rect = self.background1.image.get_rect(topleft=(0, 0))
        self.background1.rect_copy = self.background1.image.get_rect(topleft=(self.background1.rect.topright[0] + 1, self.background1.rect.topright[1] + 1))

        # Regressor stuffs
        self.regressor._prediction_score_threshold = np.random.randint(5, 10, 1)[0]
        self.regressor.regressor_warning.set_animation_state(False)
        self.regressor.regressor_warning.set_current_frame(0)
        self._mode = np.random.randint(0, 2, 1)[0]
        self.regressor.set_mode(mode=self._mode)
        if self._mode == 2:
            datastuffs.inflate_choice_file()

    def run_game(self, fade=False):
        """Runs the game."""
        # Making stuff
        self.background1.set_image(random.choice(self._main_game_bg))
        self.background2.set_image(random.choice(self._loss_bg))
        mover = Block(self._width//2, self._height - 200, self._BLOCK_SIZE, self._GREY)

        # In case that we want to transition to the game via the fade
        if fade:
            self._fade = True
            self.overlay.set_mode("fadeout")

        # Event loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Handling movement
                mover.handle_movement(event)

                # Detecting the block within tile
                for box in self.boxes:
                    box.detection(event, mover._rect, self.background1, self._scoretxt,
                                  self.regressor)
                    
            # Move tutorial text
            self._tutorial_text._rect.x = self._tutorial_text._rect.x - 4
            if self._tutorial_text._rect.x < -3125:
                self._tutorial_text._rect.x = self._width

            # Regressor events
            # Awaiting for the value to meet its threshold
            self.regressor.awaiting_prediction()

            # To get the red overlay on:
            # If a prediction is being made, but the prediction is not done,
            # nor is the overlay being animated,
            # nor is the overlay on,
            # then put the overlay on
            if self.regressor._predicting and not self.regressor._ready_to_predict and not self.regressor.warning_overlay._done and not self.readyred:
                # print("ACTIVATE OVERLAY")
                self.readyred = True

            # If the regressor threshold has been met...
            if self.regressor._ready_to_predict:
                prediction = self.regressor.compare_results()
                # print("MADE PREDICTION")
                if prediction:
                    self.loss = True
                    # BUG FIX - Ensures no score is given for wrong predictions
                    Initialisation.score -= Box.box_choices[len(Box.box_choices) - 1]

            # Drawing the stuffs - ordering functions allows for layering
            if not self.loss:
                self._screen.fill(self._BLACK)
                self.background1.update()
                self._scoretxt.update_self()
                self.boxes.update()
                mover.draw()
                # Only draw if boolean says so
                if not self.disable_tutorial_text:
                    self._tutorial_text.update_self()
                self.regressor_group.update()

            # Upon loss
            else:
               self.lose_game()

            # Making the red overlay
            if self.readyred and not self.regressor.warning_overlay._done:
                # print("PLAY OVERLAY")
                self.regressor.warning_overlay.update()
            elif self.readyred and self.regressor.warning_overlay._done:
                # print("WARNING OVERLAY DONE")
                self.readyred = False
            
            # Fading the screen out to make a smooth transition
            if self._fade and not self.overlay._done:
                self.overlay.update()

            if self._fade and self.overlay._done:
                self.overlay._done = False
                self._fade = False

            # Updating game state
            pygame.display.update()
            self._clock.tick(self._FPS)

    def handle_toggle_text(self):
        """Handle the ON/OFF message for the 'toggle tutorial' text"""
        self.disable_tutorial_text = not self.disable_tutorial_text
        if not self.disable_tutorial_text:
            self._instruct_on_losstxt_2.set_text("Press T to toggle tutorial text [ON]")
        else:
            self._instruct_on_losstxt_2.set_text("Press T to toggle tutorial text [OFF]")

    def lose_game(self):
        """Method run on the loss of the game."""
        # Re-defining loss score text for making score proper
        self._loss_scoretxt = Text(f"Score: {Initialisation.score}", self._width//2, self._height//2.5, 50, self._WHITE, font_choice=0)

        # Bool to handle fade in/out
        fadein = False

        # Handling the high-score list and the high score display
        df = pd.read_csv(join('Data', 'highscores.csv'))
        df_copy = df.copy()
        for i in range(3):
            if df.at[i,'Score'] < Initialisation.score:
                # Placing each value down the score list
                j = i
                while j < 2:
                    j += 1
                    df.at[j, 'Score'] = df_copy.at[j - 1, 'Score']
                df.at[i,'Score'] = Initialisation.score
                break
        self.highscores_df = df
        df.to_csv(join('Data', 'highscores.csv'), index=False)
        self._highscore_txt.set_text(f"Top Scores: {self.highscores_df.iat[0, 0]}, {self.highscores_df.iat[1, 0]}, {self.highscores_df.iat[2, 0]}")
        self._highscore_txt.rect = self._highscore_txt._text.get_rect(
            center=(self._highscore_txt._x, self._highscore_txt._y)) # Encapsulation illegal

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                # Reset game if R key is pressed
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        fadein = True
                        self.overlay.set_mode("fadein")
                    if event.key == pygame.K_t:
                        # print("PRESSED T")
                        self.handle_toggle_text()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.r_key.rect.collidepoint(event.pos):
                            fadein = True
                            self.overlay.set_mode("fadein")
                        if self.t_key.rect.collidepoint(event.pos):
                            # print("PRESSED T")
                            self.handle_toggle_text()

            # Animates the keys
            if not self.r_key.get_animation_state():
                self.r_key.set_animation_state(True)
            
            if not self.t_key.get_animation_state():
                self.t_key.set_animation_state(True)

            if self.overlay._done:
                self.reset_stats()
                self.run_game(fade=True)
            
            self.background2.update()
            self._highscore_txt.update_self()
            self._loss_scoretxt.update_self()
            self._losstxt.update_self()
            self._instruct_on_losstxt_2.update_self()
            self._instruct_on_losstxt.update_self()
            self.t_key.update()
            self.r_key.update()

            if fadein:
                self.overlay.update()

            pygame.display.update()
            self._clock.tick(self._FPS)

class Block(Initialisation):
    def __init__(self, x: int, y: int, size: int, colour: tuple | list):
        """Create a moving block."""
        super().__init__()
        self._x = x 
        self._y = y
        self._size = size
        self._rect = pygame.Rect(self._x, self._y, self._size, self._size)
        self._colour = list(colour)
        self._original_colour = self._colour
        self._pressed = False

        # Animation
        self._curr_colour = 0
        self._rainbow = False

    def set_colour(self, colour: tuple | list):
        """Set the colour."""
        self._colour = list(colour)

    def rainbow_colour(self, phase, frequency):
        """Sets the tuple for the colour of the block."""
        # Phase - offset value for sine function
        # Frequency - rate of change for sine function
        r = np.sin(frequency * phase) * 127 + 128
        g = np.sin(frequency * phase + 2 * np.pi / 3) * 127 + 128
        b = np.sin(frequency * phase + 4 * np.pi / 3) * 127 + 128
        return (int(r), int(g), int(b))

    def draw(self):
        """Draw the block."""
        if not self._rainbow:
            pygame.draw.rect(self._screen, self._colour, self._rect)
        else:
            if self._curr_colour > 100000:
                self._curr_colour = 0
            pygame.draw.rect(self._screen, self.rainbow_colour(self._curr_colour, 20), self._rect)

    def contain(self):
        """Contains the block."""
        x = self._rect.midleft[0]
        y = self._rect.top

        # Checking the bounds of the width of the block, ensuring the body does not go out of bounds
        # To the left
        if not 0 < x:
            x = 1
        if not x < self._width - self._size:
            x = self._width - self._size - 1

        # Doing same for the height of the block
        if not 0 < y:
            y = 1
        if not y < self._height - self._size:
            y = self._height - self._size - 1

        # Setting the values
        self._rect = pygame.Rect(x, y, self._size, self._size)

    def handle_movement(self, event):
        """Handle the movement of the block."""
        # Mouse button is down
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self._rect.collidepoint(event.pos):
                    self._pressed = True
                    self._colour = (24, 48, 155)
                    self._rainbow = True

        # Mouse button is up
        if event.type == pygame.MOUSEBUTTONUP:
            self._pressed = False
            self._colour = self._original_colour
            self._rainbow = False

        # Mouse is moving
        if event.type == pygame.MOUSEMOTION:
            if self._pressed:
                self._curr_colour += 0.01
                self._rect.move_ip(event.rel[0], event.rel[1])
                self.contain()

class Box(pygame.sprite.Sprite, Initialisation):
    boxes = []
    box_choices = []
    bg_speed_ref = 1
    def __init__(self, x: int, y: int, image, choice: tuple):
        """
        The box where the block will be placed in.

        Each tile will be 64 by 64 pixels wide.

        Choice -> (x, y, width, height)
        """
        pygame.sprite.Sprite.__init__(self)  # Initialize the sprite
        Initialisation.__init__(self)  # Initialize the Initialisation class
        Box.boxes.append(np.random.randint(0, 1))

        # Main variables
        self._x = x 
        self._y = y
        self.image = pygame.transform.scale(get_sprite_sheet(image, choice), (self._BLOCK_SIZE + 50, self._BLOCK_SIZE + 50))
        self.rect = self.image.get_rect(topleft=(self._x, self._y))
        self._id = len(Box.boxes)

        # Detection
        self.detection_offset = 60
        self.detection_rect = pygame.Rect(self._x + self.detection_offset, self._y + self.detection_offset,
                                          30, 30)

    def show_detection_zone(self):
        """Test method to show detection zone."""
        pygame.draw.rect(self._screen, self._WHITE, self.detection_rect)

    def detection(self, event, rect, bg, score_text, regressor):
        """Detects if it has been chosen.""" 
        # Mouse button is up
        if event.type == pygame.MOUSEBUTTONUP:
            if self.detection_rect.collidepoint(rect.center) and self.rect.collidepoint(event.pos):
                # Detect that it has been chosen
                Box.box_choices.append(self._id)
                # Manipulate choice file
                datastuffs.write_to_choice_file([self._id])
                # Update score
                Initialisation.score += self._id
                score_text.set_text(f"Score: {Initialisation.score}")
                # Tell regressor about what was chosen
                if regressor._predicting:
                    regressor.actual_value = self._id
                    regressor._ready_to_predict = True
                # Speed the background up every 1 /(that number) choices
                Box.bg_speed_ref += 0.33
                bg.set_speed(np.floor(Box.bg_speed_ref))

    def update(self):
        """Update the sprite."""
        self._screen.blit(self.image, self.rect)

# Decor Classes
class Background(pygame.sprite.Sprite, Initialisation):
    def __init__(self, path: str, width: int, height: int, direction: int = 0):
        """Creates a background."""
        pygame.sprite.Sprite.__init__(self)  # Initialize the sprite
        Initialisation.__init__(self)  # Initialize the Initialisation class

        # Stuff
        self._width = width
        self._height = height
        self.image_path = path
        self.image_normal = pygame.image.load(self.image_path).convert_alpha()
        self.image = pygame.transform.scale(self.image_normal, (width, height))
        self.image_flipped = pygame.transform.flip(self.image, flip_x=True, flip_y=False)
        self._speed = 1
        self._direction = direction
        self.set_speed(1) # To add that negative change if needed

        # Rectangles
        self.rect = self.image.get_rect(topleft=(0, 0))
        # BUG FIX - removed +- 1 from topleft specification to stop from seeing the black screen
        if direction == 0:
            self.rect_copy = self.image.get_rect(topleft=(self.rect.topright[0], self.rect.topright[1]))
        else:
            self.rect_copy = self.image.get_rect(topright=(self.rect.topleft[0], self.rect.topleft[1]))

    def set_image(self, path):
        """Sets the background image."""
        self.image_path = path
        self.image_normal = pygame.image.load(self.image_path).convert_alpha()
        self.image = pygame.transform.scale(self.image_normal, (self._width, self._height))
        self.image_flipped = pygame.transform.flip(self.image, True, False)

    def get_speed(self):
        """Get your speed."""
        return self._speed
    
    def set_speed(self, speed):
        """Set the speed of the background."""
        if self._direction == 0:
            self._speed = speed
        else:
            self._speed = -speed

    def update(self):
        """Update the state of the background."""
        # Move the backgrounds
        self.rect.x -= self._speed
        self.rect_copy.x -= self._speed

        # Moving left
        if self._direction == 0:
        # Wrapping around the background, so that it can stay on screen permanently
            if self.rect.midright[0] <= abs(self._speed):
                self.rect.topleft = (self._width - abs(self._speed), 0)
            
            self._screen.blit(self.image, self.rect)

            if self.rect_copy.midright[0] <= abs(self._speed):
                self.rect_copy.topleft = (self._width - abs(self._speed), 0)

            self._screen.blit(self.image_flipped, self.rect_copy)

        # Moving right
        else:
            if self.rect.midleft[0] >= self._width:
                self.rect.topright = (abs(self._speed), 0)
            
            self._screen.blit(self.image, self.rect)

            if self.rect_copy.midleft[0] >= self._width:
                self.rect_copy.topright = (abs(self._speed), 0)

            self._screen.blit(self.image_flipped, self.rect_copy)

class Image(pygame.sprite.Sprite, Initialisation):
    def __init__(self, path, x, y, width, height, transparent=False):
        """Creates a simple image from the given path."""
        pygame.sprite.Sprite.__init__(self)
        Initialisation.__init__(self)

        # Basic stuff
        self._path = path
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._transparent = transparent
        self.image = pygame.transform.scale(pygame.image.load(self._path).convert_alpha(), (width, height))
        self.rect = self.image.get_rect(topleft=(x,y))

    def update(self):
        """Updates the image."""
        if not self._transparent:
            self._screen.blit(self.image, self.rect)
        else:
            # i dont know what this does and you never will need to know
            temp_surface = pygame.Surface(self._transparency_layer.get_size(), pygame.SRCALPHA)
            temp_surface.blit(self.image, (0, 0))
            self._transparency_layer.blit(temp_surface, self.rect)
            
class AnimatedImage(pygame.sprite.Sprite, Initialisation):
    def __init__(self, path: str, width_frame: int, height_frame: int, start_coordinates: tuple, 
                 pos_x: int, pos_y: int, size: int, frames: int = 1, interval: float = (1/3)):
        """Create an image to be used, with possible frames."""
        # Setup
        pygame.sprite.Sprite.__init__(self)
        Initialisation.__init__(self)
        self._path = path
        self._frames = list()
        self.index = 0
        self._size = size

        # Getting each frame of the image
        for frame in range(frames):
            x = start_coordinates[0] + (width_frame * frame)
            y = start_coordinates[1]
            self._frames.append(get_sprite_sheet(path=self._path, frame=(x, y, width_frame, height_frame)))
    
        self._image_temp = self._frames[self.index]
        self.image = pygame.transform.scale(self._image_temp, (self._size, self._size))
        self._x = pos_x
        self._y = pos_y
        self.rect = self.image.get_rect(center=(self._x, self._y))

        # Animation variables
        self._animate = False
        self._interval = interval
        self._current_frame = 0
        self._target_frame = len(self._frames) * 2

    def get_animation_state(self):
        """Returns if the image is currently animating."""
        return self._animate
    
    def set_animation_state(self, state: bool):
        """Sets the state of animation."""
        self._animate = state

    def set_current_frame(self, frame: float):
        """Sets the current frame of animation."""
        self._current_frame = frame

    def set_image(self, index: float):
        """Sets the image correctly."""
        self._image_temp = self._frames[int(index)].convert_alpha()
        self.image = pygame.transform.scale(self._image_temp, (self._size, self._size))

    def update(self):
        """Update the image."""
        # Animate using frames
        if self._animate:
            self._current_frame += self._interval

            # Blit the image
            # BUG FIX - removed else so that the image does not temporarily disappear (added min)
            self.set_image(min(len(self._frames) -1, self._current_frame))
            self._screen.blit(self.image, self.rect)

            # Reset it
            if self._current_frame >= len(self._frames):
                self.set_animation_state(False)
                self._current_frame = 0
                
class Text(Initialisation):
    all_text = []

    def __init__(self, contents: str, x: int, y: int, size: int, colour: tuple, font_choice: int = 0):
        """
        A customisable Text class that piggybacks off the existing Pygame text class.
        
        Text Choices:\n
        - 0 -> Origa\n
        - 1 -> Pixellari
        """
        super().__init__()
        font_choices = {
            0: join("Assets", "Fonts", "origa___.ttf"),
            1: join("Assets", "Fonts", "Pixellari.ttf")
        }

        main_font = pygame.font.Font(font_choices[font_choice], size) # Font object, font can be used commercially
        self._font = main_font
        self._contents = contents
        self._x = x
        self._y = y
        self._colour = colour
        self._text = self._font.render(contents, True, colour)
        self._rect = self._text.get_rect(center = (x, y))
        Text.all_text.append(self)

    def set_text(self, text):
        """Sets the new message of the text."""
        self._contents = text
        self._text = self._font.render(self._contents, True, self._colour)

    def set_colour(self, colour):
        """Sets the colour of the text."""
        self._text = self._font.render(self._contents, True, colour)

    def update_self(self):
        """Displays the text onto the current screen."""
        self._screen.blit(self._text, self._rect)

    @classmethod
    def update_all(cls):
        """Display ALL text onto the current screen."""
        for text in Text.all_text:
            text.update_self()

class Overlay(Initialisation):
    def __init__(self, width: int, height: int, min_opacity: int, max_opacity: int, rate: float, colour: tuple | pygame.Color, mode: str = "fadein"):
        """Initialises an overlay surface, that can be used as a screen to fade in and out.
        
        There are two modes:
        - "fadein" --> allows for the fading in of the overlay
        - "fadeout" --> allows for the fading out of the overlay
        - "pulse" --> a mixture of fade in and fade out

        """
        super().__init__()
        self._width = width
        self._height = height
        self._min_opacity, self._max_opacity = (min_opacity, max_opacity)
        self._rate = rate
        self._opacity = 0
        self._colour = colour
        self._mode = mode.lower()
        self._done = False
        self._rebound = False # Only for 'pulse'

        self._overlay = pygame.Surface((self._width, self._height), pygame.SRCALPHA)
        self._rect = self._overlay.get_rect(topleft=(0, 0))
        self._overlay.fill(colour)

    def reset(self):
        """Resets the overlay to be used again."""
        self._opacity = 0
        self._done = False
        self._rebound = False

    def set_rate(self, rate):
        self._rate = rate
        
    def set_mode(self, mode):
        """Sets the mode, as well as resetting certain attributes."""
        self._mode = mode.lower()
        self._done = False
        self._rebound = False

    def update(self):
        """Updates the overlay."""
        # Fading in
        if self._mode == 'fadein':
            if self._opacity < self._max_opacity:
                self._done = False
                self._opacity += self._rate
                # Sets the alpha/transparency of the surface
                # and keeps the opacity within its range
                # random comma amidst scary brackets so a tuple is made
                self._overlay.fill(self._colour + (max(0, min(255, int(self._opacity,))),))
            else:
                self._done = True
        
        # Fading out, the reverse of fade in
        elif self._mode == 'fadeout':
            if self._opacity > self._min_opacity:
                self._done = False
                self._opacity -= self._rate
                self._overlay.fill(self._colour + (min(255, max(0, int(self._opacity))),))
            else:
                self._done = True

        # Pulse
        # Fades in and when done, fades out
        elif self._mode == 'pulse':
            if self._opacity < self._max_opacity and not self._rebound:
                self._done = False
                self._opacity += self._rate
                self._overlay.fill(self._colour + (max(0, min(255, int(self._opacity,))),))
            elif self._opacity >= self._min_opacity and not self._rebound:
                self._rebound = True
            
            if self._opacity > self._min_opacity and self._rebound:
                self._opacity -= self._rate
                self._overlay.fill(self._colour + (min(255, max(0, int(self._opacity))),))
            elif self._opacity >= self._min_opacity and self._rebound:
                self._done = True

        self._screen.blit(self._overlay, self._rect)

# Predictor Classes
class Regressor(Initialisation):
    def __init__(self, mode):
        """
        This is the class for the machine that tries to predict what a human
        will do.

        ### Modes
        - 0 (Statistical Warfare): Predicts based upon the mean or the mode, depending on what is more preferrable
        - 1 (Linear Devastation): Predicts based upon a Linear Regression model
        - 2 (Logistic Adjudication): Predicts based upon a Logistic Regression model

        """
        super().__init__()
        # Main attributes
        self._mode = mode
        self._df = pd.read_csv(join('Data', 'choices.csv'))
        self._mode_list = {
            0: self.statistical_warfare,
            1: self.linear_devastation,
            2: self.logistic_adjudication
        }

        # Prediction time
        self._prediction_score_threshold = np.random.randint(5, 10, 1)[0]
        self._predicting = False
        self._ready_to_predict = False
        self.predicted_value = -1
        self.actual_value = -1

        # Image-related things
        self.regressor_warning = AnimatedImage(
            join("Assets", "Images", "fx1.png"), 64, 64, (0,320), self._width//2, self._height // 6,
            256, frames=9
        )
        self.warning_overlay = Overlay(self._width, self._height, min_opacity=0, max_opacity=60, rate=3, 
                                     colour=(224, 61, 16), mode='pulse')
    
    def update_dataframe(self):
        """Updates the dataframe the regressor has."""
        self._df = pd.read_csv(join('Data', 'choices.csv'))

    def set_threshold(self):
        """Resets the threshold to something meaningful."""
        self._prediction_score_threshold = Initialisation.score + np.random.randint(1, 10, 1)[0]
    
    def reset(self):
        """Resets all variables needed for resetting."""
        self.set_threshold()
        self._predicting = False
        self._ready_to_predict = False
        self.predicted_value = -1
        self.actual_value = -1
    
    def compare_results(self):
        """Compares the results of the values it has on it."""
        result = self.predicted_value == self.actual_value
        self.reset()
        return result

    def awaiting_prediction(self):
        """Watches for the conditions for a prediction to be made."""
        if Initialisation.score >= self._prediction_score_threshold and not self._predicting:
            self._predicting = True
            self.warning_overlay.reset()
            self.update_dataframe()
            # Ensures animation always plays
            self.regressor_warning.set_current_frame(0)
            self.regressor_warning.set_animation_state(state=True)
            self._mode_list[self._mode]()

    def statistical_warfare(self):
        """Predicts values based upon the mean and the mode."""
        # gets the mean and mode values, as well as the number of times the most common number appears
        df_means = self._df.values.mean()
        df_mode = self._df.mode().iat[0, 0]
        
        # gonna predict the next value
        self.predicted_value = min(int(round((df_means + df_mode) / 2)), 4)

    def linear_devastation(self):
        """Creates an equation of form y = a + bx to predict the next value."""
        # get variables
        t = self._df.index.values
        T = t.reshape(-1, 1) # matrix form
        y = self._df.values
        # create a model
        model = LinearRegression()
        model.fit(T, y)
        # lambda expression for equation
        # Minmax to ensure that the number stays between 1 and 4
        f = lambda x: max(1, min(4, round(model.intercept_[0] + model.coef_[0][0] * x)))

        # gonna predict the next value
        self.predicted_value = f(len(t))
    
    def logistic_adjudication(self):
        """Aims to predict the next value using a logistic regression model."""
        X = self._df.index.values.reshape(-1, 1)
        left = False # Checking if we are looking at left values on number line
        y = pd.Series([(val > 2) for val in self._df['Box'].values]) # Binary searching
        # Model
        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(X, y)
        a = model.intercept_
        b = model.coef_
        f = lambda x: round((1/(1+np.exp(-(a + b * x)))[0][0]), 4) # Logistic eauation
        prob = f(len(X)) # P(val>2)

        # Next binary iter
        if prob < 0.5:
            y = pd.Series([(val < 2) for val in self._df['Box'].values])
            left = True
            # P(val = 1)
        else:
            y = pd.Series([(val > 3) for val in self._df['Box'].values])
            # P(val = 4)

        # Recalibrating model
        model.fit(X, y)
        a = model.intercept_
        b = model.coef_
        f = lambda x: round((1/(1+np.exp(-(a + b * x)))[0][0]), 4)
        prob = f(len(X) + 1)

        # If P(val=1) is being looked at
        if left:
            if prob < 0.5:
                self.predicted_value = 2
            else:
                self.predicted_value = 1
        # If P(val=4) is being looked at
        else:
            if prob < 0.5:
                self.predicted_value = 3
            else:
                self.predicted_value = 4

    def set_mode(self, mode):
        """Sets the mode of the Regressor."""
        self._mode = mode

if __name__ == "__main__":
    game = Uchinogasu()
    game.run_game()