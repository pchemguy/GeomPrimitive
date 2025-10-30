def test_draw_adds_line_to_axes(fig_ax, line_instance):
  fig, ax = fig_ax
  before = len(ax.lines)
  line_instance.reset(ax).draw(ax)
  after = len(ax.lines)
  assert after == before + 1
