import pygame

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Update Text Example")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the font
font = pygame.font.Font(None, 36)

# Initialize the variable and text
player_name = "Player"
text = f"Welcome, {player_name}!"
text_surface = font.render(text, True, WHITE)
text_rect = text_surface.get_rect()
text_rect.midtop = (200, 50)

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player_name = "Guest"
                text = f"Welcome, {player_name}!"
                text_surface = font.render(text, True, WHITE)

    # Clear the screen
    screen.fill(BLACK)

    # Draw the text
    screen.blit(text_surface, text_rect)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()