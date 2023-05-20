import licorice_font

class InfoMessage:
    def __init__(self, INFO="INFO", color="BLUE"):
        self.INFO = licorice_font.font_format(INFO, [color])

    def _format_msg_title(self, INFO):
        return f" - [{INFO}] | "

    def __call__(self, msg):

        self.BASE = self._format_msg_title(self.INFO)
        print(self.BASE + msg)