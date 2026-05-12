
def format_accuracy(accuracy, p=4):

    f = '{:.' + str(p) + 'f}'
    s = '1.' + '0' * p

    if f.format(accuracy) == s and accuracy < 1.0:
        return s + '~'
    else:
        return f.format(accuracy)

def format_loss(loss, p=4):

    f = '{:.' + str(p) + 'f}'
    s = '0.' + '0' * p

    if f.format(loss) == s and loss > 0.0:
        return s + '~'
    else:
        return f.format(loss)

