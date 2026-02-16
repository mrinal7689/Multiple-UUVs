from entity_draw import entity_draw


def draw_world(world, screen):
    for h in world.hunters:
        entity_draw(h, screen)

    for t in world.targets:
        entity_draw(t, screen)
