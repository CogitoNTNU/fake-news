
import thorpy, pygame

class MyGame(object):

    def __init__(self):
        #a counter displaying the trials and some hint/infos about the game
        self.textBox = thorpy.make_text(text="",
                                          font_color=(0,0,0),
                                          font_size=20,)
        self.textBox.set_font("HELVETICA")
        #the inserter element in which player can insert his guess
        self.insertBox = thorpy.Inserter(name="")
        self.insertBox.set_size((200,20))
        self.background = thorpy.Background( image="BlankTrumpTweet.jpg",
                                                    elements=[self.insertBox,
                                                              self.textBox])
        thorpy.store(self.background, gap=20)
        #reaction called each time the player has inserted something
        insertReaction = thorpy.ConstantReaction(
                            reacts_to=thorpy.constants.THORPY_EVENT,
                            reac_func=self.insertReactionFunction,
                            event_args={"id":thorpy.constants.EVENT_INSERT,
                                        "el":self.insertBox})
        self.background.add_reaction(insertReaction)


    def insertReactionFunction(self): #here is all the dynamics of the game
        value = self.insertBox.get_value() #get text inserted by player
        new_text = makePrediction(value)
        self.insertBox.set_value("") #wathever happens, we flush the inserter
        self.insertBox.unblit_and_reblit() #redraw inserter
        self.textBox.unblit()
        self.textBox.update()
        self.textBox.set_text(new_text)
        self.textBox.center(axis=(True,False)) #center on screen only x-axis
        self.textBox.blit()
        self.textBox.update()
        self.insertBox.enter() #inserter keeps the focus


    def launchGame(self):
        self.insertBox.enter() #giv the focus to inserter
        menu = thorpy.Menu(self.background) #create and launch the menu
        menu.play()
    
def makePrediction(inputText):
    return inputText
application = thorpy.Application(size=(1200, 630), caption="Trump tweet simulator")
mygame = MyGame()
mygame.launchGame()
application.quit()