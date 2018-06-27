'''
Script takes in the .txt files and convert them to parquet for easy reading
'''

# TODO


def convert_labels(file):
    '''
    function that takes in the .txt files from labels and convert them to
    usable numpy arrays
    '''
    numbers_we_care = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    temp = []
    for i in file.readlines():
        for j in i:
            if j in numbers_we_care:
                temp.append(int(j))
    temp = np.array(temp)
    return temp


if __name__ == '__main__':
    pass
