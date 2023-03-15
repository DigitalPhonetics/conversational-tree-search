
class Topic(object):
    DIALOG_START = 'dialog_start'  # Called at the beginning of a new dialog. Subscribe here to set stateful variables for one dialog.
    DIALOG_END = 'dialog_end'      # Called at the end of a dialog (after a bye-action).
    DIALOG_EXIT = 'dialog_exit'    # Called when the dialog system shuts down. Subscribe here if you e.g. have to close resource handles / free locks.
