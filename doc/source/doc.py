# See https://michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html

# # -- Fixing bug with google docs showing attributes-------------
from sphinx.ext.napoleon.docstring import GoogleDocstring

# first, we define new methods for any new sections and add them to the class
def parse_keys_section(self, section):
    return self._format_fields('Keys', self._consume_fields())
GoogleDocstring._parse_keys_section = parse_keys_section

def parse_attributes_section(self, section):
    return self._format_fields('Attributes', self._consume_fields())
GoogleDocstring._parse_attributes_section = parse_attributes_section

def parse_class_attributes_section(self, section):
    return self._format_fields('Class Attributes', self._consume_fields())
GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section

# we now patch the parse method to guarantee that the the above methods are
# assigned to the _section dict
def patched_parse(self):
    self._sections['keys'] = self._parse_keys_section
    self._sections['class attributes'] = self._parse_class_attributes_section
    self._unpatched_parse()
GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse