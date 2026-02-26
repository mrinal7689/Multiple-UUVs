import pygame
from settings import WIDTH, HEIGHT, FPS, TEXT_COLOR
from world import MultiTargetWorld


def draw_hud(screen, world, clock):
    font = pygame.font.Font(None, 24)
    fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, TEXT_COLOR)
    hunters_text = font.render(f"Hunters: {len(world.hunters)}", True, TEXT_COLOR)
    targets_text = font.render(f"Targets: {len(world.targets)}", True, TEXT_COLOR)
    captures_text = font.render(f"Captures: {world.captures}", True, TEXT_COLOR)

    
    screen.blit(hunters_text, (10, 35))
    screen.blit(targets_text, (10, 60))
    screen.blit(captures_text, (10, 85))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Multi-UUV Hunting Simulation")
    clock = pygame.time.Clock()

    world = MultiTargetWorld()
    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        running = world.step()

        world.draw(screen)
        draw_hud(screen, world, clock)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()