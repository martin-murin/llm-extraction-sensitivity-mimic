-- Normalize wide supplement tables to stable wrapped column widths.
-- This avoids LaTeX overflow from long identifiers and code/pattern lists.

local function lower(s)
  return string.lower(s or "")
end

local function header_cells(tbl)
  if not tbl.head or not tbl.head.rows or #tbl.head.rows == 0 then
    return {}
  end
  local row = tbl.head.rows[1]
  local cells = {}
  for i, cell in ipairs(row.cells or {}) do
    cells[i] = lower(pandoc.utils.stringify(cell))
  end
  return cells
end

local function assign_widths(tbl, widths)
  if #tbl.colspecs ~= #widths then
    return tbl
  end
  local colspecs = {}
  for i = 1, #widths do
    colspecs[i] = { pandoc.AlignLeft, widths[i] }
  end
  tbl.colspecs = colspecs
  return tbl
end

function Table(tbl)
  if not tbl.colspecs or #tbl.colspecs == 0 then
    return tbl
  end

  local h = header_cells(tbl)
  local h1 = h[1] or ""
  local h2 = h[2] or ""
  local h3 = h[3] or ""

  -- S6 sample design
  if h1:find("sample") and h2:find("n") then
    return assign_widths(tbl, { 0.20, 0.08, 0.42, 0.30 })
  end

  -- S2 ICD LF table (4 columns)
  if h1:find("lf") and h2:find("target") and #tbl.colspecs == 4 then
    return assign_widths(tbl, { 0.24, 0.30, 0.16, 0.30 })
  end

  -- S3 regex LF table (3 columns)
  if h1:find("lf") and h2:find("target") and #tbl.colspecs == 3 then
    return assign_widths(tbl, { 0.40, 0.30, 0.30 })
  end

  -- S1 tag vocabulary
  if h1:find("tag") and h3:find("anchor") then
    return assign_widths(tbl, { 0.30, 0.40, 0.30 })
  end

  -- S5 TriState definitions
  if h1:find("field") and h2:find("definition") and #tbl.colspecs == 2 then
    return assign_widths(tbl, { 0.38, 0.62 })
  end

  -- S3 enum fields
  if h1:find("field") and h2:find("value") then
    return assign_widths(tbl, { 0.30, 0.18, 0.52 })
  end

  return tbl
end
