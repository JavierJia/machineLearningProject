function ds = dsig(x)
% derivative of (scaled) sigmoid
ds = sig(z) .* (1-sig(z));