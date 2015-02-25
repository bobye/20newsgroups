function v = nmi(label, result)
%% Copyright (c) 2010, Michael Chen
%% All rights reserved.

%% Redistribution and use in source and binary forms, with or without
%% modification, are permitted provided that the following conditions are
%% met:

%%     * Redistributions of source code must retain the above copyright
%%       notice, this list of conditions and the following disclaimer.
%%     * Redistributions in binary form must reproduce the above copyright
%%       notice, this list of conditions and the following disclaimer in
%%       the documentation and/or other materials provided with the distribution

%% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%% POSSIBILITY OF SUCH DAMAGE.

% Nomalized mutual information
% Written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
assert(length(label) == length(result));

label = label(:);
result = result(:);

n = length(label);

label_unique = unique(label);
result_unique = unique(result);

% check the integrity of result
%if length(label_unique) ~= length(result_unique)
%    error('The clustering result is not consistent with label.');
%end;

cl = length(label_unique);
cr = length(result_unique);

% distribution of result and label
Ml = double(repmat(label,1,cl) == repmat(label_unique',n,1));
Mr = double(repmat(result,1,cr) == repmat(result_unique',n,1));
Pl = sum(Ml)/n;
Pr = sum(Mr)/n;

% entropy of Pr and Pl
Hl = -sum( Pl .* log2( Pl + eps ) );
Hr = -sum( Pr .* log2( Pr + eps ) );


% joint entropy of Pr and Pl
% M = zeros(c);
% for I = 1:c
% 	for J = 1:c
% 		M(I,J) = sum(result==result_unique(I)&label==label_unique(J));
% 	end;
% end;
% M = M / n;
M = Ml'*Mr/n;
Hlr = -sum( M(:) .* log2( M(:) + eps ) );

% mutual information
MI = Hl + Hr - Hlr;

% normalized mutual information
v = MI/max(Hl, Hr);



