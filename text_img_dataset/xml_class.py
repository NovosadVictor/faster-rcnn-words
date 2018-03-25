### custom class for xml ###

class XML:
    def __init__(self, name, size):
        self.file = open('data/Annotations/' + name + '.xml', 'w')
        self.file.write('<annotation>\n\t<folder>data</folder>\n\t<filename>' +\
                        name + '</filename>' +\
                        '\n\t<size>\n\t\t<width>' + str(size[0]) + '</width>' +\
                        '\n\t\t<height>' + str(size[1]) + '</height>\n\t\t<depth>3</depth>\n\t</size>'
                        )

    def add_object(self, name, coordinates):
        self.file.write('\n\t<object>\n\t\t<name>' + name + '</name>' +\
                        '\n\t\t<bndbox>' +\
                        '\n\t\t\t<xmin>' + str(coordinates[0]) + '</xmin>' +\
                        '\n\t\t\t<ymin>' + str(coordinates[1]) + '</ymin>' +\
                        '\n\t\t\t<xmax>' + str(coordinates[2]) + '</xmax>' +\
                        '\n\t\t\t<ymax>' + str(coordinates[3]) + '</ymax>' +\
                        '\n\t\t</bndbox>' +\
                        '\n\t</object>'
                        )

    def save_file(self):
        self.file.write('\n</annotation>')
        self.file.close()
