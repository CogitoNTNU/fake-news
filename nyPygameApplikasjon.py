import pygame as pg

pg.init()
pg.key.set_repeat(300, 25)
screen = pg.display.set_mode((700, 750))
COLOR_INACTIVE = pg.Color('black')
COLOR_ACTIVE = pg.Color('white')
FONT = pg.font.SysFont('arial', 20)


class InputBox:

    def __init__(self, x, y, w, h, text=''):
        self.rect = pg.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = text
        self.txt_surface = FONT.render(text, True, pg.Color('black'))
        self.active = False

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    pass
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = FONT.render(self.text, True, pg.Color('black'))

    def update(self):
        # Resize the box if the text is too long.
        width = max(100, self.txt_surface.get_width() + 10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))
        # Blit the rect.
        pg.draw.rect(screen, self.color, self.rect, 2)

    def getWidth(self):
        return self.rect.w

    def getText(self):
        return self.text

class Background(pg.sprite.Sprite):
    def __init__(self, image_file, location):
        pg.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pg.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

def blit_text(surface, text, pos, font, boxWidth, color=pg.Color('black')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    x = boxWidth+20
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.



def main():
    clock = pg.time.Clock()
    input_box1 = InputBox(15, 100, 70, 30)
    input_boxes = [input_box1]

    BackGround = Background("blankTrumpTweetWComment.png", [0,0])

    tweetButton = pg.image.load('button.png')
    done = False

    text = "This is a really long sentence with a couple of breaks.\nSometimes it will break even if there isn't a break " \
           "in the sentence, but that's because the text is too long to fit the screen.\nIt can look strange sometimes.\n" \
           "This function doesn't check if the text is too high to fit on the height of the surface though, so sometimes " \
           "text will disappear underneath the surface"
    font = pg.font.SysFont('arial', 20)


    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if pg.mouse.get_pos()[0] > 550 and pg.mouse.get_pos()[0] < 650:
                        if pg.mouse.get_pos()[1] > 300 and pg.mouse.get_pos()[1] < 350:
                            text = input_box1.getText()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_CLEAR:
                    text = ''
                elif event.key == pg.K_RETURN:
                    text = input_box1.getText()
            for box in input_boxes:
                box.handle_event(event)

        for box in input_boxes:
            box.update()

        screen.fill((255, 255, 255))
        screen.blit(BackGround.image, BackGround.rect)
        screen.blit(tweetButton, (550,300))
        #print text
        blit_text(screen, text, (15, 105), font, input_box1.getWidth())
        for box in input_boxes:
            box.draw(screen)

        pg.display.flip()
        clock.tick(30)

main()
pg.quit()