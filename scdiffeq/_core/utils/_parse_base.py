from abc import ABC


class Parser:

    PASSED_KWARGS = {}
    LITERAL_KWARGS = {}
    SELF_IGNORE = ["self"]

    def __init__(
        self,
        obj,
        ignore=["self", "parser", "kwargs"],
        literal_kwargs_keys=["kwargs"],
        hidden_kwargs=[],
        set_literal_kwarg_attrs=True,
        VERBOSE=False,
    ):
        """
        Define in the __init__() step of another class to catch and organize
        passed parameters.

        Parameters:
        -----------
        kwargs

        Returns:
        --------
        None
            Sets the prescribed attributes to the recieved [obj].
        """

        self.__config__(locals())

    def __config__(self, init_passed_kwargs):

        """Setup Parser class. Organize and handle self-facing kwargs. Run on .__init__()"""

        for key, val in init_passed_kwargs.items():
            if not key in self.SELF_IGNORE:
                setattr(self, key, val)

        if self.VERBOSE:
            print(
                " - [NOTE] | SETTING THE FOLLOWING ATTRIBUTES TO: {}".format(self.obj)
            )

    def parse_literal_kwargs(self, literal_kwargs):

        """Parse kwargs encased in a dictionary that were passed using, for example: **kwargs"""

        for key, val in literal_kwargs.items():
            if not key in self.ignore:
                self.LITERAL_KWARGS[key] = val
                if self.set_literal_kwarg_attrs:
                    setattr(self.obj, key, val)

    def transfer_kwarg_dicts(self):

        """Transfer dictionaries collecting kwargs (both inferred and literal) to the passed [obj] class."""

        for key in self.__dir__():
            if key.endswith("_KWARGS"):
                setattr(self.obj, "_{}".format(key), getattr(self, key))
                if self.VERBOSE:
                    print("\t{}".format(key))

    def __call__(self, kwargs):

        """
        Define in the __init__() step of another class to catch and organize
        passed parameters.

        Parameters:
        -----------
        kwargs

        Returns:
        --------
        None
            Sets the prescribed attributes to the recieved class.

        Examples:
        ---------
        >>> class SomeClass:
        >>>     def __init__(self, x=2, y=3, **kwargs):
        >>>         parser = Parser(self)
        >>>         parser(locals())
        """

        for n, (key, val) in enumerate(kwargs.items()):

            if not key in self.ignore:
                self.PASSED_KWARGS[key] = val
                if key in self.hidden_kwargs:
                    key = "_{}".format(key)
                setattr(self.obj, key, val)
                if self.VERBOSE:
                    print("\t{}".format(key))
            elif key in self.literal_kwargs_keys:
                self.parse_literal_kwargs(val)

        self.transfer_kwarg_dicts()
        

class ParseBase(ABC):
    """Base class for automatic parsing of args."""

    def __parse__(
        self,
        kwargs,
        ignore=["self", "parser", "kwargs", "__class__"],
        literal_kwargs_keys=["kwargs"],
        hidden_kwargs=[],
        set_literal_kwarg_attrs=True,
        VERBOSE=False,
    ):

        p = Parser(
            obj=self,
            ignore=ignore,
            literal_kwargs_keys=literal_kwargs_keys,
            hidden_kwargs=hidden_kwargs,
            set_literal_kwarg_attrs=set_literal_kwarg_attrs,
            VERBOSE=VERBOSE,
        )
        p(kwargs)