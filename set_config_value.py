"""Set a dotted config value in a saved config.py, INSERTING the line if it is absent.

sed cannot do the insert case, and that case matters: evaluation.py reads
config.env.test_size unguarded, so a config that simply lacks the line dies with
AttributeError only after the whole environment has spun up.

Usage:  python3 set_config_value.py <config.py> env.test_size 3500
        python3 set_config_value.py <config.py> training.output_dir "'data/exp_heightpub'"
"""
import re
import sys


def main(path, dotted, value):
    src = open(path).read()
    pat = r'(?m)^(\s*%s\s*=\s*).*$' % re.escape(dotted)
    if re.search(pat, src):
        src = re.sub(pat, lambda m: m.group(1) + value, src, count=1)
        action = 'set'
    else:
        # Anchor the insert to a line that must exist in the same section, keeping the
        # surrounding indentation so the assignment stays inside the Config class body.
        section = dotted.split('.')[0]
        anchor = re.search(r'(?m)^(\s*)%s\.[a-z_]+\s*=.*$' % re.escape(section), src)
        if anchor:
            indent = anchor.group(1)
            src = src[:anchor.end()] + '\n%s%s = %s' % (indent, dotted, value) + src[anchor.end():]
            action = 'INSERTED (was missing)'
        else:
            # The whole SECTION is absent (e.g. a `cbf` block in a config snapshot written
            # before that feature existed). Create it as a BaseConfig() right after the
            # last line of any existing section, so saved configs from older runs can be
            # extended without hand-editing.
            last = None
            for m in re.finditer(r'(?m)^(\s*)[a-z_]+\.[a-z_]+\s*=.*$', src):
                last = m
            if last is None:
                print('FAILED: cannot find any <section>.<key> line to anchor to')
                return 1
            indent = last.group(1)
            src = (src[:last.end()]
                   + '\n\n%s%s = BaseConfig()' % (indent, section)
                   + '\n%s%s = %s' % (indent, dotted, value)
                   + src[last.end():])
            action = 'SECTION CREATED + INSERTED'
    open(path, 'w').write(src)
    print('%s: %s = %s' % (action, dotted, value))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
