
# -- import packages: ----------------------------------------------------------
import licorice_font


# -- set typing: ---------------------------------------------------------------
NoneType = type(None)


class InfoMessage:
    def __init__(
        self, silent: bool = False, INFO: str = "INFO", color: str = "BLUE"
    ) -> NoneType:

        """

        Parameters:
        -----------
        silent
            type: bool
            default: False

        INFO
            type: str
            default: "INFO"

        color
            type: str
            default: "BLUE"

        Returns:
        --------
        None
        """

        self.INFO = licorice_font.font_format(INFO, [color])
        self._SILENT = silent

    def _format_msg_title(self, INFO) -> str:
        return f" - [{INFO}] | "

    def __call__(self, msg: str) -> NoneType:

        """
        Prints message according to above-specified formatting.

        Parameters:
        -----------
        msg
            type: str
        """

        if not self._SILENT:
            self.BASE = self._format_msg_title(self.INFO)
            print(self.BASE + msg)
