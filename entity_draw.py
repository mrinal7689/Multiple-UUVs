import pygame


def entity_draw(entity, screen):
    pygame.draw.circle(
        screen,
        entity.color,
        (int(entity.pos[0]), int(entity.pos[1])),
        entity.size,
    )
