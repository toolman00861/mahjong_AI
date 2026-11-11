#!/usr/bin/env python3
# Patch Mortal libriichi: tonpuu (East-only) support + obs version 5 (v4 + tonpuu/all-last rows) + arena game_length
import re,sys,pathlib
R=pathlib.Path(sys.argv[1])/'libriichi'/'src'
def sub(path, old, new, count=1):
    p=R/path; s=p.read_text()
    n=s.count(old)
    assert n>=1, f'{path}: pattern not found: {old[:60]!r}'
    if count==1: assert n==1, f'{path}: pattern ambiguous ({n}): {old[:60]!r}'
    s=s.replace(old,new) if count!=1 else s.replace(old,new,1)
    p.write_text(s); print('patched',path)

# 1. event: optional tonpuu flag on start_kyoku
sub('mjai/event.rs','''        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
    },''','''        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
        /// Extension: Some(true) for tonpuu (East-only) games.
        #[serde(default)]
        tonpuu: Option<bool>,
    },''')
# 2. player state field
sub('state/player_state.rs','''    pub(super) is_all_last: bool,''','''    pub(super) is_all_last: bool,
    /// Extension: East-only game.
    pub(super) is_tonpuu: bool,''')
# 3. update.rs
sub('state/update.rs','''                scores,
                tehais,
            } => self.start_kyoku(
                bakaze,
                dora_marker,
                kyoku,
                honba,
                kyotaku,
                oya,
                scores,
                tehais,
            )?,''','''                scores,
                tehais,
                tonpuu,
            } => self.start_kyoku(
                bakaze,
                dora_marker,
                kyoku,
                honba,
                kyotaku,
                oya,
                scores,
                tehais,
                tonpuu.unwrap_or(false),
            )?,''')
sub('state/update.rs','''        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
    ) -> Result<()> {''','''        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
        tonpuu: bool,
    ) -> Result<()> {''')
sub('state/update.rs','''        self.is_all_last = match self.bakaze.as_u8() {
            tu8!(E) => false,
            tu8!(S) => self.kyoku == 3,
            _ => true,
        };''','''        self.is_tonpuu = tonpuu;
        self.is_all_last = match self.bakaze.as_u8() {
            tu8!(E) => tonpuu && self.kyoku == 3,
            tu8!(S) => tonpuu || self.kyoku == 3,
            _ => true,
        };''')
# 4. board: tonpuu flag
sub('arena/board.rs','''    /// Goes forward (iter)
    pub ura_indicators: Vec<Tile>,
}''','''    /// Goes forward (iter)
    pub ura_indicators: Vec<Tile>,
    /// Extension: East-only game.
    pub tonpuu: bool,
}''')
sub('arena/board.rs','''            scores: self.board.scores,
            tehais: self.board.haipai,
        };''','''            scores: self.board.scores,
            tehais: self.board.haipai,
            tonpuu: self.board.tonpuu.then_some(true),
        };''')
sub('arena/board.rs','2 | 3 | 4 => {','2 | 3 | 4 | 5 | 6 => {')
sub('dataset/invisible.rs','2 | 3 | 4 => {','2 | 3 | 4 | 5 | 6 => {')
# 5. game.rs: propagate length
sub('arena/game.rs','''                scores: self.scores,
                ..Default::default()
            };
            next_board.init_from_seed(self.seed);''','''                scores: self.scores,
                tonpuu: self.length == 4,
                ..Default::default()
            };
            next_board.init_from_seed(self.seed);''')
# 6. one_vs_three: game_length
sub('arena/one_vs_three.rs','''    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
}''','''    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
    pub game_length: u8,
}''')
sub('arena/one_vs_three.rs','''    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
        }
    }''','''    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None, game_length=8))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>, game_length: u8) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
            game_length,
        }
    }''')
sub('arena/one_vs_three.rs','''        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);''','''        let batch_game = BatchGame {
            length: self.game_length,
            init_scores: [25000; 4],
            disable_progress_bar: self.disable_progress_bar,
        };''')
# 7. consts
sub('consts.rs','pub const MAX_VERSION: u32 = 4;','pub const MAX_VERSION: u32 = 6;')
sub('consts.rs','        4 => (1012, 34),','        4 => (1012, 34),\n        5 | 6 => (1014, 34),')
sub('consts.rs','        2 | 3 | 4 => (217, 34),','        2 | 3 | 4 | 5 | 6 => (217, 34),')
# 8. obs_repr: v5 = v4 + 2 rows
sub('state/obs_repr.rs','''    at_kan_select: bool,
    version: u32,
}''','''    at_kan_select: bool,
    version: u32,
    extra_v5: bool,
    skip_sp: bool,
}''')
sub('state/obs_repr.rs','''        assert!(version <= MAX_VERSION);
        let shape = obs_shape(version);''','''        assert!(version <= MAX_VERSION);
        let shape = obs_shape(version);
        let extra_v5 = version == 5 || version == 6;
        let skip_sp = version == 6;
        let version = if extra_v5 { 4 } else { version };''')
sub('state/obs_repr.rs','''            at_kan_select,
            version,
        }''','''            at_kan_select,
            version,
            extra_v5,
            skip_sp,
        }''')
sub('state/obs_repr.rs','''        assert_eq!(self.idx, self.arr.rows());
        let arr = self.arr.build();''','''        if self.extra_v5 {
            if state.is_tonpuu {
                self.arr.fill(self.idx, 1.);
            }
            self.idx += 1;
            if state.is_all_last {
                self.arr.fill(self.idx, 1.);
            }
            self.idx += 1;
        }

        assert_eq!(self.idx, self.arr.rows());
        let arr = self.arr.build();''')
sub('state/test.rs','ps.test_update(&Event::StartKyoku {','ps.test_update(&Event::StartKyoku {\n        tonpuu: None,', count=2)
sub('state/obs_repr.rs','''            if let Ok(SinglePlayerTables { max_ev_table }) = state.single_player_tables() {''','''            let sp_tables = if self.skip_sp { None } else { state.single_player_tables().ok() };
            if let Some(SinglePlayerTables { max_ev_table }) = sp_tables {''')
sub('state/player_state.rs','''    /// Raises an exception if the action is not valid.
    #[pyo3(name = "validate_reaction")]''','''    /// Extension: agari points (ron, tsumo_ko, tsumo_oya) without ura dora.
    #[pyo3(name = "agari_points")]
    fn agari_points_py(&self, is_ron: bool) -> Result<(i32, i32, i32)> {
        let p = self.agari_points(is_ron, &[])?;
        Ok((p.ron, p.tsumo_ko, p.tsumo_oya))
    }

    /// Raises an exception if the action is not valid.
    #[pyo3(name = "validate_reaction")]''')
print('ALL PATCHES OK')