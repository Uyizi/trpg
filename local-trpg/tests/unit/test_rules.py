import unittest
from local_trpg.engine.rules import classify_check,Engine,initial_state
from local_trpg.schemas import ActionIntent
from local_trpg.modules import load_module
class Rules(unittest.TestCase):
 def test_thresholds(self):
  self.assertEqual([classify_check(x) for x in [6,7,10]],['failure','success_with_cost','success'])
 def test_exact_tick12_restore_survives_but_two_tick_crossing_does_not(self):
  m=load_module();s=initial_state(m,'测试');s.tick=11;s.flags['cargo_branch_isolated']=True;s.actors['player'].location_id='engine'
  e=Engine(s,m,'t',lambda k:[1,2]);e.resolve_player(ActionIntent(actor_id='player',kind='REPAIR',module_action_id='restore_main_loop'));e.finish_clock()
  self.assertTrue(e.state.flags['main_loop_restored']);self.assertFalse(e.state.flags['engine_destroyed'])
  s=initial_state(m,'测试');s.tick=11;e=Engine(s,m,'t',lambda k:[1,2]);e.advance(2);e.finish_clock()
  self.assertTrue(e.state.flags['engine_destroyed'])
 def test_cells_expiry_cannot_refresh(self):
  m=load_module();s=initial_state(m,'测试');s.actors['player'].location_id='engine'
  e=Engine(s,m,'t',lambda k:[6,6]);e.resolve_player(ActionIntent(actor_id='player',kind='REPAIR',module_action_id='isolate_branch'));e.finish_clock()
  expiry=e.state.cells[0]['expiry'];e.module_action('isolate_branch','player',[])
  self.assertEqual(e.state.cells[0]['expiry'],expiry)
